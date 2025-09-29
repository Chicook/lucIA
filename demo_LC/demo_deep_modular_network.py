import logging

from models.neural.modules import DeepNetworkModulesBuilder, generate_pyramid_specs


def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    # Config de ejemplo: 12 capas ocultas en pirámide desde 1024 neuronas
    input_size = 128
    output_size = 4
    specs = generate_pyramid_specs(depth=12, top_width=1024, activation='relu', dropout=0.3, batch_norm=True, decay=0.7)

    builder = DeepNetworkModulesBuilder(input_size=input_size, output_size=output_size, specs=specs, output_activation='softmax')

    # Construir, mostrar resumen y registrar/guardar
    net = builder.build()
    print(net.get_summary())

    model_id = builder.create_and_register(name="DeepModularPyramid_128x4_d12_w1024")
    print("Modelo registrado:", model_id)


if __name__ == '__main__':
    main()


