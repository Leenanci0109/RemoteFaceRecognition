// SRT.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <srt.h>
#include <vector>
#include <string>
#include <fstream>
#pragma comment(lib, "ws2_32.lib")

using namespace std;
int main()
{
    //starting setup
    srt_startup();

    addrinfo hints;
    addrinfo* res;
    //setting ip and port
    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_flags = AI_PASSIVE;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;

    string service("9000");
   

    if (0 != getaddrinfo(NULL, service.c_str(), &hints, &res))
    {
        cout << "port error\n" << endl;
        return 0;
    }

    SRTSOCKET serv = srt_create_socket();

    SRT_TRANSTYPE tt = SRTT_FILE;
    srt_setsockopt(serv, 0, SRTO_TRANSTYPE, &tt, sizeof tt);

#ifdef _WIN32
    int mss = 1052;
    srt_setsockopt(serv, 0, SRTO_MSS, &mss, sizeof(int));
#endif
    //bind to port
    if (SRT_ERROR == srt_bind(serv, res->ai_addr, res->ai_addrlen))
    {
        cout << "bind error: " << srt_getlasterror_str() << endl;
        return 0;
    }

    freeaddrinfo(res);

    cout << "connection ready at " << service << endl;
    srt_listen(serv, 10);

    sockaddr_storage clientaddr;
    int addrlen = sizeof(clientaddr);
    //accept connection
    SRTSOCKET fhandle;
    if (SRT_INVALID_SOCK == (fhandle = srt_accept(serv, (sockaddr*)&clientaddr, &addrlen)))
    {
        cout << "accept: " << srt_getlasterror_str() << endl;
        return 0;
    }

    char clienthost[NI_MAXHOST];
    char clientservice[NI_MAXSERV];
    getnameinfo((sockaddr*)&clientaddr, addrlen, clienthost, sizeof(clienthost), clientservice, sizeof(clientservice), NI_NUMERICHOST | NI_NUMERICSERV);
    cout << "new connection: " << clienthost << ":" << clientservice << endl;

    const char* path = "D:/sendpic.jpg";
    // get file size
    fstream ifs(path, ios::in | ios::binary);
    ifs.seekg(0, ios::end);
    const int64_t size = ifs.tellg();
    ifs.close();

    // Send file size.
    if (SRT_ERROR == srt_send(fhandle, (char*)&size, sizeof(int64_t)))
    {
        cout << "send: " << srt_getlasterror_str() << endl;
        return 0;
    }

    SRT_TRACEBSTATS trace;
    srt_bstats(fhandle, &trace, true);

    // Sending file
    int64_t offset = 0;
    if (SRT_ERROR == srt_sendfile(fhandle, path, &offset, size, SRT_DEFAULT_SENDFILE_BLOCK))
    {
        cout << "sendfile: " << srt_getlasterror_str() << endl;
        return 0;
    }

    srt_bstats(fhandle, &trace, true);
    cout << "speed = " << trace.mbpsSendRate << "Mbits/sec" << endl;
    const int64_t losspercent = 100 * trace.pktSndLossTotal / trace.pktSent;
    cout << "network loss = " << trace.pktSndLossTotal << "pkts (" << losspercent << "%)\n";

    srt_close(fhandle);


    srt_close(serv);

   
    srt_cleanup();

    return 0;
}