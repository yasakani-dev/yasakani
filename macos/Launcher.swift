// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
import AppKit
import Darwin

/// A small native frontend for the bundled CLI. It never invokes a shell and
/// never installs Python packages or downloads a model on the user's behalf.
final class LauncherDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    private var window: NSWindow!
    private let modelField = NSTextField(string: "")
    private let portField = NSTextField(string: "8765")
    private let contextField = NSTextField(string: "4096")
    private let statusField = NSTextField(labelWithString: "Stopped")
    private let endpointField = NSTextField(labelWithString: "")
    private let logView = NSTextView()
    private let chooseButton = NSButton(title: "Choose…", target: nil, action: nil)
    private let startButton = NSButton(title: "Start server", target: nil, action: nil)
    private let stopButton = NSButton(title: "Stop", target: nil, action: nil)
    private let chatButton = NSButton(title: "Open chat", target: nil, action: nil)
    private var process: Process?
    private var outputPipe: Pipe?
    private var outputBuffer = Data()
    private var healthTimer: Timer?
    private var healthTask: URLSessionDataTask?
    private var serverURL: URL?
    private var stopping = false
    private var quitting = false
    private var sawReadyLog = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        let menu = NSMenu()
        let applicationItem = NSMenuItem()
        menu.addItem(applicationItem)
        let applicationMenu = NSMenu()
        applicationItem.submenu = applicationMenu
        applicationMenu.addItem(withTitle: "About YASAKANI", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        applicationMenu.addItem(.separator())
        applicationMenu.addItem(withTitle: "Quit YASAKANI", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        let editItem = NSMenuItem()
        editItem.title = "Edit"
        let editMenu = NSMenu(title: "Edit")
        for (title, selector, key) in [("Cut", #selector(NSText.cut(_:)), "x"),
                                       ("Copy", #selector(NSText.copy(_:)), "c"),
                                       ("Paste", #selector(NSText.paste(_:)), "v"),
                                       ("Select All", #selector(NSText.selectAll(_:)), "a")] {
            editMenu.addItem(withTitle: title, action: selector, keyEquivalent: key)
        }
        editItem.submenu = editMenu
        menu.addItem(editItem)
        NSApp.mainMenu = menu

        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 760, height: 560),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        window.title = "YASAKANI"
        window.minSize = NSSize(width: 620, height: 440)
        window.delegate = self
        window.isReleasedWhenClosed = false
        let heading = NSTextField(labelWithString: "YASAKANI for Mac")
        heading.font = .boldSystemFont(ofSize: 23)
        let description = NSTextField(wrappingLabelWithString: "Choose a local model, then start the server. The API and chat are available only on this Mac.")
        description.textColor = .secondaryLabelColor

        let defaults = UserDefaults.standard
        modelField.stringValue = defaults.string(forKey: "modelDirectory") ?? ""
        modelField.placeholderString = "Model folder"
        modelField.setAccessibilityLabel("Model folder")
        portField.stringValue = defaults.string(forKey: "port") ?? "8765"
        portField.setAccessibilityLabel("Server port")
        contextField.stringValue = defaults.string(forKey: "context") ?? "4096"
        contextField.setAccessibilityLabel("Context length in tokens")
        chooseButton.target = self
        chooseButton.action = #selector(chooseModel)
        startButton.target = self
        startButton.action = #selector(startServer)
        startButton.keyEquivalent = "\r"
        stopButton.target = self
        stopButton.action = #selector(stopServer)
        chatButton.target = self
        chatButton.action = #selector(openChat)
        let modelRow = NSStackView(views: [modelField, chooseButton])
        modelRow.spacing = 8
        let optionsRow = NSStackView(views: [NSTextField(labelWithString: "Port"), portField,
                                           NSTextField(labelWithString: "Context tokens"), contextField])
        optionsRow.spacing = 10
        portField.widthAnchor.constraint(equalToConstant: 80).isActive = true
        contextField.widthAnchor.constraint(equalToConstant: 100).isActive = true
        let buttons = NSStackView(views: [startButton, stopButton, chatButton])
        buttons.spacing = 8
        endpointField.isSelectable = true
        endpointField.textColor = .secondaryLabelColor
        endpointField.font = .monospacedSystemFont(ofSize: 12, weight: .regular)

        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.borderType = .bezelBorder
        logView.isEditable = false
        logView.isSelectable = true
        logView.isRichText = false
        logView.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
        logView.textContainerInset = NSSize(width: 8, height: 8)
        logView.autoresizingMask = [.width]
        logView.isVerticallyResizable = true
        logView.isHorizontallyResizable = false
        logView.textContainer?.widthTracksTextView = true
        scroll.documentView = logView
        scroll.heightAnchor.constraint(greaterThanOrEqualToConstant: 150).isActive = true
        let stack = NSStackView(views: [heading, description, modelRow, optionsRow, buttons, statusField, endpointField, scroll])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 12
        stack.translatesAutoresizingMaskIntoConstraints = false
        let content = window.contentView!
        content.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 24),
            stack.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -24),
            stack.topAnchor.constraint(equalTo: content.topAnchor, constant: 24),
            stack.bottomAnchor.constraint(equalTo: content.bottomAnchor, constant: -24),
            modelRow.widthAnchor.constraint(equalTo: stack.widthAnchor),
            description.widthAnchor.constraint(equalTo: stack.widthAnchor),
            scroll.widthAnchor.constraint(equalTo: stack.widthAnchor)
        ])
        updateControls()
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func chooseModel() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose model folder"
        panel.beginSheetModal(for: window) { [weak self] response in
            if response == .OK, let url = panel.url { self?.modelField.stringValue = url.path }
        }
    }

    @objc private func startServer() {
        guard process == nil else { return }
        let path = (modelField.stringValue as NSString).expandingTildeInPath
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return reportError("Choose an existing model folder.")
        }
        guard let port = Int(portField.stringValue), (1024...65535).contains(port) else {
            return reportError("The port must be a whole number from 1024 to 65535.")
        }
        guard let context = Int(contextField.stringValue), (256...262144).contains(context) else {
            return reportError("The context length must be a whole number from 256 to 262144.")
        }
        guard let resources = Bundle.main.resourceURL else { return reportError("This app has no Resources folder.") }
        let executable = resources.appendingPathComponent("runtime/yasakani")
        guard FileManager.default.isExecutableFile(atPath: executable.path) else {
            return reportError("The bundled runtime is missing. Please use the complete YASAKANI.app distribution.")
        }
        let child = Process()
        child.executableURL = executable
        child.arguments = ["serve", "--backend", "metal", "--model-dir", path, "--host", "127.0.0.1", "--port", String(port),
                           "--max-ctx", String(context), "--web-ui", resources.appendingPathComponent("chat.html").path]
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        // The launcher has no API-key UI. Do not silently inherit a key that
        // would make the bundled browser chat unusable.
        environment.removeValue(forKey: "YASAKANI_API_KEY")
        child.environment = environment
        let pipe = Pipe()
        child.standardOutput = pipe
        child.standardError = pipe
        child.standardInput = FileHandle.nullDevice
        outputBuffer.removeAll(keepingCapacity: true)
        sawReadyLog = false
        stopping = false
        serverURL = URL(string: "http://127.0.0.1:\(port)")
        pipe.fileHandleForReading.readabilityHandler = { [weak self, weak child] handle in
            let data = handle.availableData
            if data.isEmpty { handle.readabilityHandler = nil }
            DispatchQueue.main.async {
                guard let self, let child, self.process === child else { return }
                self.consumeOutput(data)
            }
        }
        child.terminationHandler = { [weak self] child in
            DispatchQueue.main.async { self?.didExit(child) }
        }
        process = child
        outputPipe = pipe
        statusField.stringValue = "Starting — loading the model…"
        endpointField.stringValue = "API: http://127.0.0.1:\(port)/v1"
        appendLog("Starting local server…\n")
        updateControls()
        do {
            try child.run()
            let defaults = UserDefaults.standard
            defaults.set(path, forKey: "modelDirectory")
            defaults.set(String(port), forKey: "port")
            defaults.set(String(context), forKey: "context")
            healthTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in self?.checkHealth() }
        } catch {
            pipe.fileHandleForReading.readabilityHandler = nil
            process = nil
            outputPipe = nil
            statusField.stringValue = "Could not start"
            updateControls()
            reportError("Could not start the bundled runtime: \(error.localizedDescription)")
        }
    }

    private func consumeOutput(_ data: Data) {
        outputBuffer.append(data)
        while let newline = outputBuffer.firstIndex(of: 10) {
            let line = String(decoding: outputBuffer[...newline], as: UTF8.self)
            outputBuffer.removeSubrange(...newline)
            // Only our own child's readiness message permits a ready status.
            // An unrelated server on an occupied port cannot satisfy this.
            if line.contains("ready: OpenAI-compatible API at ") { sawReadyLog = true }
            appendLog(line)
        }
        if outputBuffer.count > 65536 || data.isEmpty {
            appendLog(String(decoding: outputBuffer, as: UTF8.self))
            outputBuffer.removeAll(keepingCapacity: true)
        }
    }

    private func checkHealth() {
        guard let child = process, child.isRunning, !stopping, healthTask == nil, let serverURL else { return }
        var request = URLRequest(url: serverURL.appendingPathComponent("health"))
        request.timeoutInterval = 2
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let task = URLSession.shared.dataTask(with: request) { [weak self, weak child] data, response, _ in
            DispatchQueue.main.async {
                guard let self, let child, self.process === child, !self.stopping else { return }
                self.healthTask = nil
                let body = data.flatMap { try? JSONSerialization.jsonObject(with: $0) } as? [String: Any]
                let ready = (response as? HTTPURLResponse)?.statusCode == 200 && body?["status"] as? String == "ok" && self.sawReadyLog
                self.statusField.stringValue = ready ? "Ready — server running on this Mac" : "Starting — loading the model…"
                self.chatButton.isEnabled = ready
            }
        }
        healthTask = task
        task.resume()
    }

    @objc private func openChat() {
        if chatButton.isEnabled, let serverURL { NSWorkspace.shared.open(serverURL) }
    }

    @objc private func stopServer() {
        guard let child = process, !stopping else { return }
        stopping = true
        statusField.stringValue = "Stopping…"
        cancelHealthChecks()
        updateControls()
        if child.isRunning { child.terminate() }
        // Model loading or an active native operation may delay SIGTERM. Keep
        // the app alive until this exact child exits, then reclaim it.
        DispatchQueue.main.asyncAfter(deadline: .now() + 5) { [weak self, weak child] in
            guard let self, let child, self.process === child, child.isRunning else { return }
            self.appendLog("The server did not stop within 5 seconds; terminating it.\n")
            kill(child.processIdentifier, SIGKILL)
        }
    }

    private func didExit(_ child: Process) {
        guard process === child else { return }
        outputPipe?.fileHandleForReading.readabilityHandler = nil
        // EOF callbacks can be queued after the termination handler. Read the
        // remaining bytes before releasing the pipe so startup errors survive.
        if let data = try? outputPipe?.fileHandleForReading.readToEnd() { consumeOutput(data) }
        consumeOutput(Data())
        cancelHealthChecks()
        appendLog("Server exited (status \(child.terminationStatus)).\n")
        statusField.stringValue = stopping || child.terminationStatus == 0 ? "Stopped" : "Server failed — see log below"
        process = nil
        outputPipe = nil
        serverURL = nil
        sawReadyLog = false
        stopping = false
        updateControls()
        if quitting { NSApp.reply(toApplicationShouldTerminate: true) }
    }

    private func cancelHealthChecks() {
        healthTimer?.invalidate()
        healthTimer = nil
        healthTask?.cancel()
        healthTask = nil
        chatButton.isEnabled = false
    }

    private func updateControls() {
        let running = process != nil
        startButton.isEnabled = !running
        stopButton.isEnabled = running && !stopping
        chooseButton.isEnabled = !running
        modelField.isEnabled = !running
        portField.isEnabled = !running
        contextField.isEnabled = !running
        if !running || stopping { chatButton.isEnabled = false }
    }

    private func appendLog(_ text: String) {
        guard !text.isEmpty else { return }
        let attributes: [NSAttributedString.Key: Any] = [.font: NSFont.monospacedSystemFont(ofSize: 12, weight: .regular),
                                                       .foregroundColor: NSColor.textColor]
        logView.textStorage?.append(NSAttributedString(string: text, attributes: attributes))
        if let storage = logView.textStorage, storage.length > 1_048_576 {
            storage.deleteCharacters(in: NSRange(location: 0, length: storage.length - 786_432))
        }
        logView.scrollToEndOfDocument(nil)
    }

    private func reportError(_ message: String) {
        appendLog(message + "\n")
        let alert = NSAlert()
        alert.messageText = "YASAKANI could not start"
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.beginSheetModal(for: window)
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard process != nil else { return .terminateNow }
        quitting = true
        stopServer()
        return .terminateLater
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

let application = NSApplication.shared
let delegate = LauncherDelegate()
application.setActivationPolicy(.regular)
application.delegate = delegate
application.run()
