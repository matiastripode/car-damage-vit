import CoreML
import UIKit

/// Clasificador on-device que usa el modelo CoreML generado por exportar_coreml.py.
/// Devuelve el mismo tipo `Prediccion` que `APIService`, por lo que `ResultadoView`
/// funciona sin modificaciones.
struct LocalClassifier {
    private let model: CarDamageClassifier

    init() throws {
        let config = MLModelConfiguration()
        #if targetEnvironment(simulator)
        config.computeUnits = .cpuOnly
        #else
        config.computeUnits = .all
        #endif
        model = try CarDamageClassifier(configuration: config)
    }

    func predecir(imagen: UIImage) throws -> Prediccion {
        guard let cgImage = imagen.cgImage else {
            throw URLError(.cannotDecodeContentData)
        }

        let input = try CarDamageClassifierInput(pixel_valuesWith: cgImage)
        let output = try model.prediction(input: input)

        let clase = output.classLabel
        // Busca el MultiArray de logits (type=5) y aplica softmax manual.
        // ClassifierConfig de coremltools no convierte los valores a probabilidades
        // cuando el modelo viene de un wrapper trazado.
        let clases = ["dent", "scratch", "crack", "glass_shatter", "tire_flat", "lamp_broken", "fondo"]
        let logitsFeature = output.featureNames.first {
            output.featureValue(for: $0)?.type == .multiArray
        }
        let probs: [String: Double]
        if let fname = logitsFeature,
           let multiArray = output.featureValue(for: fname)?.multiArrayValue {
            let logits = (0..<clases.count).map { Double(truncating: multiArray[$0]) }
            let expLogits = logits.map { exp($0) }
            let sumExp = expLogits.reduce(0, +)
            let softmax = expLogits.map { $0 / sumExp }
            probs = Dictionary(uniqueKeysWithValues: zip(clases, softmax))
        } else {
            probs = [:]
        }
        let confianza = probs[clase] ?? 0

        let top3 = probs
            .sorted { $0.value > $1.value }
            .prefix(3)
            .map { PrediccionItem(clase: $0.key, confianza: $0.value) }

        // rlhfStorage es nil porque la inferencia es local (no pasa por MinIO)
        return Prediccion(clase: clase, confianza: confianza, top3: Array(top3), rlhfStorage: nil)
    }
}
