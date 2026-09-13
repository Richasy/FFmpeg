/*
 * HTTP reconnect error propagation test
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "libavformat/avio.h"
#include "libavformat/network.h"
#include "libavutil/avstring.h"
#include "libavutil/dict.h"
#include "libavutil/error.h"
#include "libavutil/thread.h"

#define IO_TIMEOUT_US 3000000
#define PARTIAL_BODY "12345678"

typedef struct HTTPReconnectServer {
    int listener;
    int result;
} HTTPReconnectServer;

static int wait_socket(int fd, int write)
{
    AVIOInterruptCB interrupt = { 0 };

    return ff_network_wait_fd_timeout(fd, write, IO_TIMEOUT_US, &interrupt);
}

static int accept_client(int listener)
{
    int client;
    int ret = wait_socket(listener, 0);

    if (ret < 0)
        return ret;

    client = accept(listener, NULL, NULL);
    if (client < 0)
        return ff_neterrno();
    if (ff_socket_nonblock(client, 1) < 0) {
        closesocket(client);
        return AVERROR(EIO);
    }
    return client;
}

static int read_request(int client, char *request, size_t capacity)
{
    size_t length = 0;

    while (length + 1 < capacity) {
        int ret = wait_socket(client, 0);
        int received;

        if (ret < 0)
            return ret;

        received = recv(client, request + length,
                        (int)(capacity - length - 1), 0);
        if (received < 0)
            return ff_neterrno();
        if (!received)
            return AVERROR_EOF;

        length += received;
        request[length] = '\0';
        if (strstr(request, "\r\n\r\n"))
            return 0;
    }

    return AVERROR(ENOSPC);
}

static int write_response(int client, const char *response)
{
    size_t remaining = strlen(response);

    while (remaining) {
        int ret = wait_socket(client, 1);
        int written;

        if (ret < 0)
            return ret;

        written = send(client, response, (int)remaining, MSG_NOSIGNAL);
        if (written < 0)
            return ff_neterrno();
        if (!written)
            return AVERROR(EIO);

        response += written;
        remaining -= written;
    }

    return 0;
}

static void *serve_http(void *opaque)
{
    static const char partial_response[] =
        "HTTP/1.1 200 OK\r\n"
        "Content-Length: 64\r\n"
        "Accept-Ranges: bytes\r\n"
        "Connection: keep-alive\r\n"
        "\r\n"
        PARTIAL_BODY;
    static const char forbidden_response[] =
        "HTTP/1.1 403 Forbidden\r\n"
        "Content-Length: 0\r\n"
        "Connection: close\r\n"
        "\r\n";
    HTTPReconnectServer *server = opaque;
    int result = 0;

    for (int request_index = 0; request_index < 2; request_index++) {
        char request[4096] = "";
        int client = accept_client(server->listener);

        if (client < 0) {
            result = client;
            break;
        }

        result = read_request(client, request, sizeof(request));
        if (result >= 0 && request_index == 1 &&
            !av_stristr(request, "\r\nRange: bytes=8-"))
            result = AVERROR_INVALIDDATA;
        if (result >= 0) {
            result = write_response(client, request_index == 0
                                    ? partial_response
                                    : forbidden_response);
        }

        closesocket(client);
        if (result < 0)
            break;
    }

    closesocket(server->listener);
    server->result = result;
    return NULL;
}

static int open_listener(int *port)
{
    struct sockaddr_in address = {
        .sin_family = AF_INET,
        .sin_addr.s_addr = htonl(INADDR_LOOPBACK),
    };
    socklen_t address_length = sizeof(address);
    int listener = ff_socket(AF_INET, SOCK_STREAM, 0, NULL);
    int ret;

    if (listener < 0)
        return listener;
    ret = ff_listen(listener, (struct sockaddr *)&address,
                    sizeof(address), NULL);
    if (ret < 0)
        goto fail;
    if (getsockname(listener, (struct sockaddr *)&address,
                    &address_length) < 0) {
        ret = ff_neterrno();
        goto fail;
    }

    *port = ntohs(address.sin_port);
    return listener;

fail:
    closesocket(listener);
    return ret;
}

int main(void)
{
    HTTPReconnectServer server = { .listener = -1 };
    AVDictionary *options = NULL;
    AVIOContext *input = NULL;
    pthread_t server_thread;
    char url[128];
    uint8_t buffer[32];
    int port;
    int ret;

    ret = ff_network_init();
    if (ret < 0)
        return 1;

    server.listener = open_listener(&port);
    if (server.listener < 0) {
        ret = server.listener;
        goto done;
    }

    ret = pthread_create(&server_thread, NULL, serve_http, &server);
    if (ret) {
        ret = AVERROR(ret);
        closesocket(server.listener);
        goto done;
    }

    snprintf(url, sizeof(url), "http://127.0.0.1:%d/stream", port);
    av_dict_set(&options, "multiple_requests", "1", 0);
    av_dict_set(&options, "reconnect", "1", 0);
    av_dict_set(&options, "reconnect_delay_max", "0", 0);
    av_dict_set(&options, "rw_timeout", "3000000", 0);
    ret = avio_open2(&input, url, AVIO_FLAG_READ, NULL, &options);
    av_dict_free(&options);
    if (ret >= 0) {
        do {
            ret = avio_read(input, buffer, sizeof(buffer));
        } while (ret > 0);
        avio_closep(&input);
    }

    pthread_join(server_thread, NULL);
    if (server.result < 0) {
        fprintf(stderr, "HTTP fixture failed: %s\n",
                av_err2str(server.result));
        ret = server.result;
    } else if (ret != AVERROR_HTTP_FORBIDDEN) {
        fprintf(stderr, "Expected HTTP 403, got %s\n", av_err2str(ret));
        ret = AVERROR_INVALIDDATA;
    } else {
        ret = 0;
    }

done:
    av_dict_free(&options);
    avio_closep(&input);
    ff_network_close();
    return ret < 0;
}
