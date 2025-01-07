import UIKit
import AVFoundation
import CoreML

class ViewController: UIViewController, UIDocumentPickerDelegate {

    @IBOutlet weak var slider: UISlider!
    @IBOutlet weak var analyzeButton: UIButton!
    @IBOutlet weak var fileNameLabel: UILabel! // 用于显示文件名
    @IBOutlet weak var predictionLabel: UILabel!
    @IBOutlet weak var recordButton: UIButton!
    
    var audioPlayer: AVAudioPlayer?
    var audioRecorder: AVAudioRecorder?
    var timer: Timer?
    var selectedFileUrl: URL?
    var isRecording = false

    // MARK: - 文件选择和播放

    @IBAction func chooseFile(_ sender: Any) {
        let supportedTypes: [UTType] = [.mp3, .wav, .aiff]
        let documentPicker = UIDocumentPickerViewController(forOpeningContentTypes: supportedTypes)
        documentPicker.delegate = self
        documentPicker.allowsMultipleSelection = false
        present(documentPicker, animated: true)
        print("Document picker presented.")
    }
    
    @IBAction func recordSound(_ sender: Any) {
        if isRecording {
            stopRecordingAndSave()
        } else {
            startRecording()
        }
    }
    @IBAction func sliderValueChanged(_ sender: UISlider) {
        // 在滑动过程中更新 UI，比如显示滑动值
        print("Slider moving: \(sender.value)")
    }
    @IBAction func sliderTouchBegan(_ sender: UISlider) {
        // 暂停播放
        audioPlayer?.pause()
        print("Slider touch began, audio paused.")
    }
    @IBAction func sliderTouchEnded(_ sender: UISlider) {
        // 只有在滑动结束时调整播放进度
        guard let audioPlayer = audioPlayer else { return }
        audioPlayer.currentTime = TimeInterval(sender.value)
        audioPlayer.play()
        print("Slider released, audio progress updated to: \(audioPlayer.currentTime)")
    }
    @IBAction func analyze(_ sender: UIButton) {
        guard let selectedUrl = selectedFileUrl else {
            showAlert(title: "Error", message: "No file selected for analysis.")
            return
        }

        guard let inputArray = preprocessAudioFile(url: selectedUrl) else {
            showAlert(title: "Error", message: "Failed to preprocess audio file.")
            return
        }

        do {
            let model = try BowelSoundClassifier()
            let prediction = try model.prediction(audioSamples: inputArray)

            // 格式化概率为百分比并显示
            if let probabilities = prediction.targetProbability as? [String: Double] {
                let resultText = probabilities.map {
                    "\($0.key): \(String(format: "%.2f", $0.value * 100))%"
                }.joined(separator: "\n")
                predictionLabel.text = resultText
            }
        } catch {
            showAlert(title: "Error", message: "Failed to analyze audio: \(error.localizedDescription)")
        }
    }
    func preprocessAudioFile(url: URL) -> MLMultiArray? {
        guard let audioFile = try? AVAudioFile(forReading: url) else { return nil }
        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount),
              let floatChannelData = buffer.floatChannelData else { return nil }

        try? audioFile.read(into: buffer)

        let channelData = floatChannelData[0]
        let dataLength = min(Int(frameCount), 15600)
        let inputArray = Array(UnsafeBufferPointer(start: channelData, count: dataLength))

        let paddedArray = inputArray + Array(repeating: 0.0, count: max(0, 15600 - inputArray.count))

        do {
            let mlArray = try MLMultiArray(shape: [15600], dataType: .float32)
            for (index, value) in paddedArray.enumerated() {
                mlArray[index] = NSNumber(value: value)
            }
            return mlArray
        } catch {
            print("Error creating MLMultiArray: \(error)")
            return nil
        }
    }

    func startRecording() {
        let audioFilename = getDocumentsDirectory().appendingPathComponent("recording.m4a")
        print("Start recording to: \(audioFilename.path)")
        
        let settings = [
            AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
            AVSampleRateKey: 44100,
            AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
        ] as [String: Any]
        
        do {
            audioRecorder = try AVAudioRecorder(url: audioFilename, settings: settings)
            //
            audioRecorder?.delegate = self
            audioRecorder?.record()
            isRecording = true
            recordButton.setTitle("Stop Recording", for: .normal)
            print("Recording started.")
        } catch {
            print("Recording error: \(error.localizedDescription)")
            showAlert(title: "Recording Error", message: "Failed to start recording: \(error.localizedDescription)")
        }
    }

    func stopRecordingAndSave() {
        guard let audioRecorder = audioRecorder else {
            print("No active audio recorder found.")
            return
        }
        
        audioRecorder.stop()
        isRecording = false
        recordButton.setTitle("Start Recording", for: .normal)
        print("Recording stopped.")
        
        let recordedUrl = audioRecorder.url // 不需要使用 if let，因为 .url 是非可选类型
        selectedFileUrl = recordedUrl
        fileNameLabel.text = recordedUrl.lastPathComponent
        print("Recording saved to: \(recordedUrl.path)")
        
        // 弹出文件保存界面
        let documentPicker = UIDocumentPickerViewController(forExporting: [recordedUrl])
        documentPicker.delegate = self
        present(documentPicker, animated: true)
        print("Document picker presented for exporting.")
    }
    func saveFileToDocuments(url: URL) -> URL? {
        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let destinationUrl = documentsDirectory.appendingPathComponent(url.lastPathComponent)
        
        // 检查文件是否可读取
        guard FileManager.default.isReadableFile(atPath: url.path) else {
            print("File is not readable: \(url.path)")
            showAlert(title: "Error", message: "Selected file is not accessible.")
            return nil
        }

        // 如果目标文件已存在，先删除
        if FileManager.default.fileExists(atPath: destinationUrl.path) {
            do {
                try FileManager.default.removeItem(at: destinationUrl)
            } catch {
                print("Failed to remove existing file: \(error.localizedDescription)")
                showAlert(title: "Error", message: "Failed to overwrite existing file.")
                return nil
            }
        }

        // 拷贝文件到沙盒目录
        do {
            try FileManager.default.copyItem(at: url, to: destinationUrl)
            print("File successfully copied to sandbox: \(destinationUrl.path)")
            return destinationUrl
        } catch {
            print("Error copying file: \(error.localizedDescription)")
            showAlert(title: "Error", message: "Failed to copy file: \(error.localizedDescription)")
            return nil
        }
    }
    func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
        guard let selectedUrl = urls.first else {
            showAlert(title: "Error", message: "No file selected.")
            return
        }

        // 开始访问安全范围资源
        if selectedUrl.startAccessingSecurityScopedResource() {
            defer { selectedUrl.stopAccessingSecurityScopedResource() }

            // 检查文件是否可读取
            guard FileManager.default.isReadableFile(atPath: selectedUrl.path) else {
                showAlert(title: "Error", message: "The selected file is not accessible.")
                print("File is not readable: \(selectedUrl.path)")
                return
            }

            // 尝试将文件复制到沙盒目录
            if let sandboxedUrl = saveFileToDocuments(url: selectedUrl) {
                self.selectedFileUrl = sandboxedUrl
                loadAndPlayAudio(from: sandboxedUrl)
                let fileName = sandboxedUrl.lastPathComponent
                fileNameLabel.text = fileName
                print("File copied to sandbox: \(sandboxedUrl.path)")
            } else {
                showAlert(title: "Error", message: "Failed to copy file to app's sandbox.")
            }
        } else {
            showAlert(title: "Error", message: "Failed to access the selected file.")
            print("Could not access security-scoped resource.")
        }
    }

    func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
        print("Document picker cancelled.")
        if let recordedUrl = selectedFileUrl {
            showAlert(title: "Save Canceled", message: "Recording saved locally at \(recordedUrl.lastPathComponent).")
        }
    }

    // MARK: - 播放音频文件
    func loadAndPlayAudio(from url: URL) {
        print("Loading audio file from: \(url.path)")
        do {
            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer?.prepareToPlay()
            audioPlayer?.play()
            print("Audio playback started.")
            
            guard let audioPlayer = audioPlayer else { return }
            slider.maximumValue = Float(audioPlayer.duration)
            slider.value = 0
            startTimer()
        } catch {
            print("Playback error: \(error.localizedDescription)")
            showAlert(title: "Error", message: "Failed to play audio: \(error.localizedDescription)")
        }
    }

    // MARK: - Timer 和实用方法

    func startTimer() {
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self = self, let player = self.audioPlayer else { return }
            self.slider.value = Float(player.currentTime)
            if player.currentTime >= player.duration { self.stopTimer() }
        }
        print("Timer started for slider updates.")
    }

    func stopTimer() {
        timer?.invalidate()
        timer = nil
        print("Timer stopped.")
    }

    func showAlert(title: String, message: String) {
        let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
        print("\(title): \(message)")
    }
    
    func getDocumentsDirectory() -> URL {
        return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }
}
extension ViewController: AVAudioRecorderDelegate {
    func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        if flag {
            print("Audio recording successfully finished.")
            showAlert(title: "Recording Saved", message: "Your recording has been saved.")
        } else {
            print("Audio recording failed.")
            showAlert(title: "Recording Error", message: "Failed to save the recording.")
        }
    }
}
