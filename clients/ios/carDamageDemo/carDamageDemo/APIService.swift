import Foundation
import UIKit

struct Prediccion: Codable {
    let clase: String
    let confianza: Double
    let top3: [PrediccionItem]
}

struct PrediccionItem: Codable {
    let clase: String
    let confianza: Double
}

private final class LocalhostSelfSignedDelegate: NSObject, URLSessionDelegate {
    func urlSession(
        _ session: URLSession,
        didReceive challenge: URLAuthenticationChallenge,
        completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void
    ) {
        guard
            challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
            let serverTrust = challenge.protectionSpace.serverTrust
        else {
            completionHandler(.performDefaultHandling, nil)
            return
        }

        let host = challenge.protectionSpace.host.lowercased()
        let allowedHosts = ["localhost", "127.0.0.1"]
        if allowedHosts.contains(host) {
            completionHandler(.useCredential, URLCredential(trust: serverTrust))
            return
        }

        completionHandler(.performDefaultHandling, nil)
    }
}

class APIService {
    // Cambiar por la URL de ngrok o IP local (ej. "http://192.168.1.42:8000")
    // static let baseURL = "https://TU-URL-NGROK.ngrok-free.app"
    static let baseURL = "https://localhost/api"
    private static let parsedBaseURL = URL(string: baseURL)

    private static let session: URLSession = {
#if DEBUG
        if let host = parsedBaseURL?.host?.lowercased(), host == "localhost" || host == "127.0.0.1" {
            return URLSession(configuration: .default, delegate: LocalhostSelfSignedDelegate(), delegateQueue: nil)
        }
#endif
        return .shared
    }()

    static func predecir(imagen: UIImage) async throws -> Prediccion {
        guard let url = URL(string: "\(baseURL)/predecir"),
              let jpegData = imagen.jpegData(compressionQuality: 0.8)
        else { throw URLError(.badURL) }

        let boundary = UUID().uuidString
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue(
            "multipart/form-data; boundary=\(boundary)",
            forHTTPHeaderField: "Content-Type"
        )

        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append(
            "Content-Disposition: form-data; name=\"archivo\"; filename=\"foto.jpg\"\r\n"
                .data(using: .utf8)!
        )
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(jpegData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        request.httpBody = body

        let (data, _) = try await session.data(for: request)
        return try JSONDecoder().decode(Prediccion.self, from: data)
    }
}
