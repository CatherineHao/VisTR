import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image
from PIL import ImageOps

def page_hinkley_test(data, delta=0.01, lambda_=15): #delta=0.005, lambda_=10, alpha=0): 
    # when the PH statistic exceeds lambda_, we consider a change point detected
    result_t = []
    length = len(data)
    PH = np.zeros(length)
    m_t = np.zeros(length)
    sum_m = 0.0
    m_t[0] = data[0]
    
    for t in range(1, length):
        # m_t[t] = alpha * m_t[t-1] + (1 - alpha) * data[t]
        m_t[t]= data[t]
        sum_m += data[t] - m_t[t-1]
        PH[t] = max(0, PH[t-1] + abs(data[t] - m_t[t-1]) - delta)
        
        if PH[t] > lambda_:
            # print(f"Change detected at index {t}, corresponding to {data_x[t]}")
            result_t.append(t)
            PH[t] = 0  # Reset after change is detected
    
    return result_t

def add_borders_to_image(image_path, output_path, border_ratio=0.15):
    with Image.open(image_path) as img:
        width, height = img.size
        print('image width: {width}, height: {height}')
        target_ratio = 16 / 9
        current_ratio = width / height

        new_width = width
        new_height = height

        if current_ratio < target_ratio:
            new_width = int(target_ratio * height)
        elif current_ratio > target_ratio:
            new_height = int(width / target_ratio)

        new_img = Image.new("RGB", (new_width, new_height), "white")
        paste_x = (new_width - width) // 2
        paste_y = (new_height - height) // 2
        new_img.paste(img, (paste_x, paste_y))

        border_width = int(width * border_ratio)
        border_height = int(height * border_ratio)

        if border_width * 9 != border_height * 16:
            border_height = int(border_width * 9 / 16)

        final_img = Image.new("RGB", (new_width + 2 * border_width, new_height + 2 * border_height), "white")
        final_img.paste(new_img, (border_width, border_height))

        final_img.save(output_path)
        # print(f"Image saved to {output_path}")

# Time series processing
file = pd.read_csv("./daily_pollutant.csv")

data_pm25 = file['pm25']
pm25 = data_pm25.values
data_o3 = file['o3']
o3 = data_o3.values
data_temp = file['temp']
temp = data_temp.values
data_rh = file['rh']
rh = data_rh.values
data_x = file['date']

PH_index_list = page_hinkley_test(data_rh)
smoothed_y = data_rh

slices_y = []
slice_index = []
start_index = 0
for cp in PH_index_list:
    data_slice = smoothed_y[start_index:cp]
    if len(data_slice) == 1:
        continue
    else:
        slices_y.append(data_slice)
        slice_index.append((start_index,cp))
        start_index = cp
    
if start_index < len(smoothed_y):
    slices_y.append(smoothed_y[start_index:])
    slice_index.append((start_index,len(smoothed_y)))

print(slice_index)
print('len:',len(slice_index))

n = len(slice_index)
plt.figure(figsize=(16,9))
print()
for i in tqdm(range(n)):
    for j in range(i,n):
        combined_slice = (slice_index[i][0], slice_index[j][1])
        # print(combined_slice)
        plt.axis('off')
        plt.plot(data_x[combined_slice[0]:combined_slice[1]], smoothed_y[combined_slice[0]:combined_slice[1]], color='blue', label='Data')
        plt.savefig('./pollutant figure/rh_{}_{}.png'.format(combined_slice[0], combined_slice[1]),format='png')
        plt.clf()


for filename in os.listdir ('./user_sketches'):
    if filename.endswith('.png'):
        filepath = os.path.join('./user_sketches', filename)
        if os.path.isfile(filepath):
            image_path = filepath
            output_path = "./norm_sketches" + filename
            add_borders_to_image(image_path, output_path)
        