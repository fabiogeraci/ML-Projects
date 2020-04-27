import numpy as np
from sklearn.preprocessing import LabelEncoder


class abalone_input:
    def encoded_Data(self):

        dataRaw = []
        DataFile = open("abalone.csv", "r")
        i = 0
        while True:
            theline = DataFile.readline()
            theline = theline.replace('\n', '')
            i=i+1

            if i>1:
                if len(theline) == 0:
                    break
                readData = theline.split(",")

                for pos in range(len(readData)):
                    if isinstance(readData[pos], str):
                        print('')
                    else:
                        readData[pos] = float(readData[pos])
                dataRaw.append(readData)

        DataFile.close()

        data = np.array(dataRaw)

        label_encoder = LabelEncoder()
        data[:, 0] = label_encoder.fit_transform(data[:, 0])
        data[:, -1] = label_encoder.fit_transform(data[:, -1])
        #print(data)
        return data
