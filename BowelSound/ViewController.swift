import UIKit
import AVFoundation
import CoreML

class ViewController: UIViewController, UIDocumentPickerDelegate {

    @IBOutlet weak var slider: UISlider!
    @IBOutlet weak var analyzeButton: UIButton!
    @IBOutlet weak var fileNameLabel: UILabel! // 用于显示文件名

    var audioPlayer: AVAudioPlayer?
    var timer: Timer?
    var selectedFileUrl: URL?

    // MARK: - 文件选择和播放

    @IBAction func chooseFile(_ sender: Any) {
        let supportedTypes: [UTType] = [.mp3, .wav, .aiff]
        let documentPicker = UIDocumentPickerViewController(forOpeningContentTypes: supportedTypes)
        documentPicker.delegate = self
        documentPicker.allowsMultipleSelection = false
        present(documentPicker, animated: true)
    }

    func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
        guard let selectedUrl = urls.first else { return }
        self.selectedFileUrl = selectedUrl
        loadAndPlayAudio(from: selectedUrl)

        // 提取文件名并显示
        let fileName = selectedUrl.lastPathComponent
        fileNameLabel.text = "Selected File: \(fileName)"
    }

    func loadAndPlayAudio(from url: URL) {
        do {
            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer?.prepareToPlay()
            audioPlayer?.play()

            guard let audioPlayer = audioPlayer else { return }
            slider.maximumValue = Float(audioPlayer.duration)
            slider.value = 0
            startTimer()
        } catch {
            showAlert(title: "Error", message: "Failed to play audio: \(error.localizedDescription)")
        }
    }
    // MARK: - Analyze Audio

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

            print("Prediction target: \(prediction.target)")
            print("Prediction probabilities: \(prediction.targetProbability)")
        } catch {
            showAlert(title: "Error", message: "Failed to analyze audio: \(error.localizedDescription)")
        }
    }

    // MARK: - Audio Preprocessing

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
    // MARK: - Timer 和实用方法

    func startTimer() {
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self = self, let player = self.audioPlayer else { return }
            self.slider.value = Float(player.currentTime)
            if player.currentTime >= player.duration { self.stopTimer() }
        }
    }

    func stopTimer() {
        timer?.invalidate()
        timer = nil
    }

    func showAlert(title: String, message: String) {
        let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
}
