//  VideoCaptureManager.swift
//  CameraTest
//
//  Created by Jun on 2024/06/08.

import AVFoundation

class VideoCaptureManager {
    
    private enum SessionSetupResult {
        case success
        case notAuthorized
        case configurationFailed
    }
    
    private enum ConfigurationError: Error {
        case cannotAddInput
        case cannotAddOutput
        case defaultDeviceNotExist
        case unsupportedZoomFactor
    }
    
    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "session.queue")
    private let videoOutputQueue = DispatchQueue(label: "video.output.queue")
    private var setupResult: SessionSetupResult = .success
    
    init() {
        sessionQueue.async {
            self.requestCameraAuthorizationIfNeeded()
            self.configureSession()
            self.startSessionIfPossible()
        }
    }
    
    private func requestCameraAuthorizationIfNeeded() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            break
        case .notDetermined:
            sessionQueue.suspend()
            AVCaptureDevice.requestAccess(for: .video, completionHandler: { granted in
                if !granted {
                    self.setupResult = .notAuthorized
                }
                self.sessionQueue.resume()
            })
        default:
            setupResult = .notAuthorized
        }
    }
    
    private func configureSession() {
        if setupResult != .success {
            return
        }
        
        session.beginConfiguration()
        
        // Set the session preset
        session.sessionPreset = .hd4K3840x2160 // Optional: adjust based on desired quality
        
        do {
            try addUltraWideVideoDeviceInputToSession()
            try addVideoOutputToSession()
            
            if let connection = session.connections.first {
                connection.videoOrientation = .portrait
            }
        } catch {
            print("error occurred: \(error.localizedDescription)")
            return
        }
        
        session.commitConfiguration()
    }
    
    private func addUltraWideVideoDeviceInputToSession() throws {
        guard let ultraWideCameraDevice = AVCaptureDevice.default(.builtInUltraWideCamera, for: .video, position: .back) else {
            print("Ultra Wide video device is unavailable.")
            setupResult = .configurationFailed
            session.commitConfiguration()
            throw ConfigurationError.defaultDeviceNotExist
        }
        
        try configureDeviceZoom(ultraWideCameraDevice, zoomFactor: 1.0) // Set to minimum zoom
        
        let videoDeviceInput = try AVCaptureDeviceInput(device: ultraWideCameraDevice)
        
        if session.canAddInput(videoDeviceInput) {
            session.addInput(videoDeviceInput)
        } else {
            setupResult = .configurationFailed
            session.commitConfiguration()
            throw ConfigurationError.cannotAddInput
        }
    }
    
    private func configureDeviceZoom(_ device: AVCaptureDevice, zoomFactor: CGFloat) throws {
        do {
            try device.lockForConfiguration()
            device.videoZoomFactor = zoomFactor // Ensure minimum zoom for maximum FOV
            device.unlockForConfiguration()
        } catch {
            throw error
        }
    }

    private func addVideoOutputToSession() throws {
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        } else {
            setupResult = .configurationFailed
            session.commitConfiguration()
            throw ConfigurationError.cannotAddOutput
        }
    }
    
    private func startSessionIfPossible() {
        switch setupResult {
        case .success:
            session.startRunning()
        case .notAuthorized:
            print("Camera usage not authorized")
        case .configurationFailed:
            print("Configuration failed")
        }
    }
    
    func setVideoOutputDelegate(with delegate: AVCaptureVideoDataOutputSampleBufferDelegate) {
        videoOutput.setSampleBufferDelegate(delegate, queue: videoOutputQueue)
    }
}
