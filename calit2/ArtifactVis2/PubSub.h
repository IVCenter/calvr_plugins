#pragma once

#include <string>
#include <iostream>
#include <stdexcept>
#include <zmq.hpp>
#include <google/protobuf/message_lite.h>
#include "protocol/packing.pb.h"
#include "snappy.h"

struct PubSocket
{
    zmq::socket_t socket;
    std::string uncompressed_buffer;
    std::string compressed_buffer;

    std::string compression;

public:

    PubSocket(
        zmq::context_t& context,
        std::string connection_string)
        :   socket(context, ZMQ_PUB), compression("")
    {
        uint64_t hwm = 1;
        socket.setsockopt(ZMQ_HWM, &hwm, sizeof(hwm));
        socket.bind(connection_string.c_str());
    }

    void set_compression(std::string compression_name)
    {
        if ((compression_name == "none")
                | (compression_name == ""))
        {
            compression = "";
            return;
        }

        if (compression_name == "snappy-1.0.5")
        {
            compression = compression_name;
            return;
        }

        throw std::logic_error("Unknown compression scheme: " + compression_name);
    }

    void send(const google::protobuf::MessageLite& msg)
    {
        msg.SerializeToString(&uncompressed_buffer);
        RemoteKinect::Packing packing;

        if (compression == "snappy-1.0.5")
        {
            size_t compressed_size = snappy::Compress(
                                         uncompressed_buffer.c_str(),
                                         uncompressed_buffer.size(),
                                         &compressed_buffer);
            packing.set_scheme("snappy-1.0.5");
            packing.set_data(compressed_buffer.c_str(), compressed_size);
            //std::cout << "Compression ratio: " <<  ((double)compressed_size)/((double)uncompressed_buffer.size()) << '\n';
        }
        else
        {
            // uncompressed
            packing.set_data(uncompressed_buffer);
        }

        zmq::message_t packet(packing.ByteSize());
        packing.SerializeToArray(packet.data(), packing.ByteSize());
        socket.send(packet);
    }
};


template<typename MsgType>
struct SubSocket
{
    zmq::socket_t socket;
    std::string   uncompressed;

public:
    SubSocket(zmq::context_t& context, std::string connection_string)
        :   socket(context, ZMQ_SUB)
    {
        uint64_t hwm = 1;
        int timeout  = 10; // in milliseconds
        socket.setsockopt(ZMQ_HWM, &hwm, sizeof(hwm));
        socket.setsockopt(ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
        socket.setsockopt(ZMQ_SUBSCRIBE, "", 0);
        socket.connect(connection_string.c_str());
    }

    bool recv(MsgType& write_to)
    {
        zmq::message_t packet;
        zmq::message_t packet_tmp;
        bool got_packet = false;

        // Try to get latest packet sent to us.
        // Even with HWM, there may be more than one packet waiting
        while (socket.recv(&packet_tmp, ZMQ_NOBLOCK))
        {
            got_packet = true;
            packet.move(&packet_tmp);
        }

        if (got_packet)
        {
            RemoteKinect::Packing packing;
            packing.ParseFromArray(
                packet.data(),
                packet.size()
            );

            if (packing.has_scheme() && packing.scheme() != "none")
            {
                if (packing.scheme() != "snappy-1.0.5")
                {
#ifndef NDEBUG
                    std::cerr << "Warning: unknown compression scheme '"
                              << packing.scheme()
                              << "'\n";
#endif
                    return false;
                }
                else
                {
                    // unpack snappy 1.0.5
                    snappy::Uncompress(
                        packing.data().c_str(),
                        packing.data().size(),
                        &uncompressed);
                    write_to.ParseFromArray(
                        uncompressed.c_str(),
                        uncompressed.size());
                    return true;
                }
            }
            else
            {
                // uncompressed packet
                write_to.ParseFromArray(
                    packing.data().c_str(),
                    packing.data().size());
                return true;
            }
        }

        return false;
    }
};

