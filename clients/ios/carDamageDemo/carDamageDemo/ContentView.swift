import PhotosUI
import SwiftUI

private let clasesDisponibles = ["dent", "scratch", "crack", "glass_shatter", "tire_flat", "lamp_broken", "fondo"]

struct ContentView: View {
    @State private var selectedItem: PhotosPickerItem?
    @State private var imagen: UIImage?
    @State private var prediccion: Prediccion?
    @State private var cargando = false
    @State private var errorMsg: String?
    @State private var corrigiendo = false
    @State private var feedbackEnviado = false
    @State private var enviandoFeedback = false
    @State private var claseSeleccionada = clasesDisponibles[0]

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                if let imagen {
                    Image(uiImage: imagen)
                        .resizable()
                        .scaledToFit()
                        .frame(maxHeight: 300)
                        .cornerRadius(12)
                }

                PhotosPicker(
                    "Seleccionar foto",
                    selection: $selectedItem,
                    matching: .images
                )
                .buttonStyle(.bordered)

                if cargando {
                    ProgressView("Clasificando...")
                } else if let p = prediccion {
                    ResultadoView(prediccion: p)
                    correctionSection(roiKey: p.rlhfStorage?.roiKey)
                } else if let e = errorMsg {
                    Text("Error: \(e)")
                        .foregroundStyle(.red)
                        .multilineTextAlignment(.center)
                }
            }
            .padding()
            .navigationTitle("Car Damage AI")
            .onChange(of: selectedItem) { _, item in
                Task {
                    guard
                        let data = try? await item?.loadTransferable(type: Data.self),
                        let ui = UIImage(data: data)
                    else { return }

                    imagen = ui
                    prediccion = nil
                    errorMsg = nil
                    cargando = true

                    do {
                        prediccion = try await APIService.predecir(imagen: ui)
                    } catch {
                        errorMsg = error.localizedDescription
                    }

                    cargando = false
                    feedbackEnviado = false
                }
            }
        }
    }

    @ViewBuilder
    private func correctionSection(roiKey: String?) -> some View {
        if feedbackEnviado {
            Text("✓ Corrección enviada")
                .foregroundStyle(.green)
                .font(.subheadline)
        } else if let roiKey {
            Button("¿Predicción incorrecta? Corregir") {
                claseSeleccionada = clasesDisponibles[0]
                corrigiendo = true
            }
            .buttonStyle(.bordered)
            .font(.subheadline)
            .sheet(isPresented: $corrigiendo) {
                correctionSheet(roiKey: roiKey)
            }
        }
    }

    @ViewBuilder
    private func correctionSheet(roiKey: String) -> some View {
        NavigationStack {
            Form {
                Section("Clase correcta") {
                    Picker("Clase", selection: $claseSeleccionada) {
                        ForEach(clasesDisponibles, id: \.self) { Text($0) }
                    }
                    .pickerStyle(.inline)
                    .labelsHidden()
                }
            }
            .navigationTitle("Corregir predicción")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancelar") { corrigiendo = false }
                }
                ToolbarItem(placement: .confirmationAction) {
                    if enviandoFeedback {
                        ProgressView()
                    } else {
                        Button("Enviar") {
                            Task {
                                enviandoFeedback = true
                                try? await APIService.enviarFeedback(roiKey: roiKey, claseCorrecta: claseSeleccionada)
                                enviandoFeedback = false
                                corrigiendo = false
                                feedbackEnviado = true
                            }
                        }
                    }
                }
            }
        }
    }
}

struct ResultadoView: View {
    let prediccion: Prediccion

    var body: some View {
        VStack(spacing: 12) {
            Text(prediccion.clase.uppercased())
                .font(.largeTitle.bold())

            Text(String(format: "Confianza: %.1f%%", prediccion.confianza * 100))
                .font(.title3)
                .foregroundStyle(.secondary)

            ProgressView(value: prediccion.confianza)
                .tint(prediccion.confianza > 0.7 ? .green : .orange)

            if !prediccion.top3.isEmpty {
                Divider()
                VStack(alignment: .leading, spacing: 4) {
                    Text("Top 3").font(.headline)
                    ForEach(prediccion.top3, id: \.clase) { item in
                        HStack {
                            Text(item.clase)
                            Spacer()
                            Text(String(format: "%.1f%%", item.confianza * 100))
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
        .padding()
        .background(.regularMaterial)
        .cornerRadius(12)
    }
}
