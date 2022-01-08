import serial
import numpy as np
from MultiLayerPerceptron import MultiLayerPerceptron

serial_port = '/dev/ttyACM0'
baud_rate = 9600
serial_obj = serial.Serial(serial_port, baud_rate)


def save_RGB_values(color, NO_samples=100):
    path = "../Data/RGB_Log_TCS34/{}_LOG_TCS34735.txt".format(color)
    with open(path, 'w+') as file:
        for i in range(NO_samples):
            line = serial_obj.readline()
            print(str(line)[2:-4])
            file.writelines(str(line)[2:-4] + '\n')
        print("Done!")
        file.close()


def main():
    colors = ['Black', 'Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Purple', 'Pink', 'White']
    color_data = np.array([[0, 0, 0, 0]])
    for color in colors:
        # save_RGB_values(color, 30)
        path = "../Data/RGB_Log_TCS34/{}_LOG_TCS34735.txt".format(color)
        arr = np.loadtxt(path, delimiter=',')
        arr = np.column_stack((arr, np.array([colors.index(color)] * arr.shape[0])))
        color_data = np.concatenate((color_data, arr))

    np.random.shuffle(color_data)

    mlp = MultiLayerPerceptron(3, [6], 9)
    mlp.train(color_data[:, :-1], color_data[:, -1])
    print("Dataset training done!")

    while True:
        try:
            sensor_input = serial_obj.readline()
            sensor_value = list(map(int, str(sensor_input)[2:-4].split(',')))
        except:
            continue
        if len(sensor_value) == 3:
            color_index = mlp.predict_classification(sensor_value)
            serial_obj.write(colors[color_index].encode())


if __name__ == "__main__":
    main()
