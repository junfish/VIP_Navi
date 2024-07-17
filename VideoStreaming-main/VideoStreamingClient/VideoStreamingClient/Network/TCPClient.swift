//
//  TCPClient.swift
//  CameraTest
//
//  Created by Jade on 2022/09/16.
//

import Foundation
import Network

class TCPClient {
    
    enum ConnectionError: Error {
        case invalidIPAddress
        case invalidPort
    }
    
    // MARK: - properties
    
    private lazy var queue = DispatchQueue(label: "tcp.client.queue")
    
    private var connection: NWConnection?
    
    private var state: NWConnection.State = .preparing
    
    // MARK: - methods
    
    func connect(to ipAddress: String, with port: UInt16) throws {
        guard let ipAddress = IPv4Address(ipAddress) else {
            throw ConnectionError.invalidIPAddress
        }
        guard let port = NWEndpoint.Port(rawValue: port) else {
            throw ConnectionError.invalidPort
        }
        let host = NWEndpoint.Host.ipv4(ipAddress)
        
        connection = NWConnection(host: host, port: port, using: .tcp)
        
        connection?.stateUpdateHandler = { [weak self] state in
            self?.state = state
            switch state {
            case .ready:
                print("Connected to server")
                self?.receiveData()  // Start receiving data once the connection is established
            default:
                break
            }
        }
        
        connection?.start(queue: queue)
    }
    
    func send(data: Data) {
        guard state == .ready else { return }
        
        connection?.send(content: data,
                         completion: .contentProcessed({ error in
            if let error = error {
                print(error)
            }
        }))
    }
    
    private func receiveData() {
        connection?.receive(minimumIncompleteLength: 1, maximumLength: 65536, completion: { [weak self] data, _, isComplete, error in
            if let data = data, !data.isEmpty {
                if let message = String(data: data, encoding: .utf8) {
                    print("Received: \(message)") // Print the received message which contains frame dimensions
                }
                if isComplete {
                    self?.connection?.cancel()
                } else {
                    self?.receiveData() // Continue receiving data
                }
            } else if let error = error {
                print("Failed to receive data: \(error)")
                self?.connection?.cancel()
            }
        })
    }
}
