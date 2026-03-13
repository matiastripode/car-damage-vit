import PhotosUI
import SwiftUI

struct ContentView: View {
    @State private var selectedItem: PhotosPickerItem?
    @State private var imagen: UIImage?
    @State private var prediccion: Prediccion?
    @State private var cargando = false
    @State private var errorMsg: String?

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
