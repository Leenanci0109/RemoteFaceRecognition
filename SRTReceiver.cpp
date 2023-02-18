

#include <iostream>
#include <srt.h>
#include <vector>
#include <string>
#include <fstream>
#pragma comment(lib, "ws2_32.lib")

using namespace std;

int main()
{
    srt_startup();

    srt_setloglevel(srt_logging::LogLevel::debug);

    struct addrinfo hints, * peer;

    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_flags = AI_PASSIVE;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;

    SRTSOCKET fhandle = srt_create_socket();
    SRT_TRANSTYPE tt = SRTT_FILE;
    srt_setsockopt(fhandle, 0, SRTO_TRANSTYPE, &tt, sizeof tt);

    if (0 != getaddrinfo("127.0.0.1", "9000", &hints, &peer))
    {
        cout << "incorrect server/peer address. " << endl;
        return -1;
    }

    // Connect to the server, implicit bind.
    if (SRT_ERROR == srt_connect(fhandle, peer->ai_addr, peer->ai_addrlen))
    {
        cout << "connect: " << srt_getlasterror_str() << endl;
        return -1;
    }

    freeaddrinfo(peer);


    // Get size information.
    int64_t size;

    if (SRT_ERROR == srt_recv(fhandle, (char*)&size, sizeof(int64_t)))
    {
        cout << "error: " << srt_getlasterror_str() << endl;
        return -1;
    }

    if (size < 0)
    {
        cout << "error\n";
        return -1;
    }

    // Receive the file.
    int64_t recvsize;
    int64_t offset = 0;

    SRT_TRACEBSTATS trace;
    srt_bstats(fhandle, &trace, true);

    if (SRT_ERROR == (recvsize = srt_recvfile(fhandle, "D:/receive1.jpg", &offset, size, SRT_DEFAULT_RECVFILE_BLOCK)))
    {
        cout << "error in receiving file : " << srt_getlasterror_str() << endl;
        return -1;
    }

    srt_bstats(fhandle, &trace, true);

    cout << "speed = " << trace.mbpsRecvRate << "Mbits/sec" << endl;
    int losspercent = 100 * trace.pktRcvLossTotal / trace.pktRecv;
    cout << "loss = " << trace.pktRcvLossTotal << "pkt (" << losspercent << "%)\n";

    srt_close(fhandle);

    // Signal to the SRT library to clean up all allocated sockets and resources.
    srt_cleanup();

    return 0;

}
