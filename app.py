# import flask
# from flask import Flask, request
import pandas as pd 
import numpy as np 
import pickle 
import xgboost as xg
import keras
from PIL import Image
from keras.models import Sequential 
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D
import cv2
import tensorflow as tf
# import flasgger 
# from flasgger import Swagger 
import streamlit as st
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt

transmission_dct = {'Automatic': 0, 'Manual': 1}
location_dct = {'Ahmedabad': 0, 'Bangalore': 1, 'Chennai': 2, 'Coimbatore': 3, 'Delhi': 4, 'Hyderabad': 5, 'Jaipur': 6, 'Kochi': 7, 'Kolkata': 8, 'Mumbai': 9, 'Pune': 10}
fuel_dct = {'CNG': 0, 'Diesel': 1, 'LPG': 2, 'Petrol': 3}
owner_dct = {'First': 0, 'Fourth & Above': 1, 'Second': 2, 'Third': 3}
brand_dct = {'audi': 0, 'bmw': 1, 'datsun': 2, 'fiat': 3, 'ford': 4, 'honda': 5, 'hyundai': 6, 'isuzu': 7, 'jaguar': 8, 'jeep': 9, 'land': 10, 'mahindra': 11, 'maruti': 12, 'mercedes-benz': 13, 'mini': 14, 'mitsubishi': 15, 'nissan': 16, 'porsche': 17, 'renault': 18, 'skoda': 19, 'tata': 20, 'toyota': 21, 'volkswagen': 22, 'volvo': 23}
model_dct = {'1': 0, '3': 1, '5': 2, '6': 3, '7': 4, '800': 5, 'a': 6, 'a-star': 7, 'a3': 8, 'a4': 9, 'a6': 10, 'a7': 11, 'a8': 12, 'accent': 13, 'accord': 14, 'alto': 15, 'amaze': 16, 'ameo': 17, 'aspire': 18, 'avventura': 19, 'b': 20, 'baleno': 21, 'bolero': 22, 'bolt': 23, 'boxster': 24, 'br-v': 25, 'brio': 26, 'brv': 27, 'c-class': 28, 'camry': 29, 'cayenne': 30, 'cayman': 31, 'cedia': 32, 'celerio': 33, 'ciaz': 34, 'city': 35, 'civic': 36, 'cla': 37, 'clubman': 38, 'compass': 39, 'cooper': 40, 'corolla': 41, 'countryman': 42, 'cr-v': 43, 'creta': 44, 'crosspolo': 45, 'duster': 46, 'dzire': 47, 'e-class': 48, 'ecosport': 49, 'eeco': 50, 'elantra': 51, 'elite': 52, 'endeavour': 53, 'eon': 54, 'ertiga': 55, 'esteem': 56, 'estilo': 57, 'etios': 58, 'evalia': 59, 'fabia': 60, 'fiesta': 61, 'figo': 62, 'fluence': 63, 'fortuner': 64, 'freestyle': 65, 'getz': 66, 'gl-class': 67, 'gla': 68, 'glc': 69, 'gle': 70, 'gls': 71, 'go': 72, 'grand': 73, 'grande': 74, 'hexa': 75, 'i10': 76, 'i20': 77, 'ignis': 78, 'ikon': 79, 'indica': 80, 'indigo': 81, 'innova': 82, 'jazz': 83, 'jeep': 84, 'jetta': 85, 'koleos': 86, 'kuv': 87, 'kwid': 88, 'laura': 89, 'linea': 90, 'lodgy': 91, 'logan': 92, 'm-class': 93, 'manza': 94, 'micra': 95, 'mobilio': 96, 'montero': 97, 'mux': 98, 'nano': 99, 'new': 100, 'nexon': 101, 'nuvosport': 102, 'octavia': 103, 'omni': 104, 'outlander': 105, 'pajero': 106, 'panamera': 107, 'passat': 108, 'petra': 109, 'platinum': 110, 'polo': 111, 'pulse': 112, 'punto': 113, 'q3': 114, 'q5': 115, 'q7': 116, 'qualis': 117, 'quanto': 118, 'r-class': 119, 'rapid': 120, 'redi-go': 121, 'renault': 122, 'ritz': 123, 'rover': 124, 'rs5': 125, 's': 126, 's-cross': 127, 's60': 128, 's80': 129, 'safari': 130, 'santa': 131, 'santro': 132, 'scala': 133, 'scorpio': 134, 'slc': 135, 'slk-class': 136, 'sonata': 137, 'ssangyong': 138, 'sumo': 139, 'sunny': 140, 'superb': 141, 'swift': 142, 'sx4': 143, 'teana': 144, 'terrano': 145, 'thar': 146, 'tiago': 147, 'tigor': 148, 'tiguan': 149, 'tt': 150, 'tucson': 151, 'tuv': 152, 'v40': 153, 'vento': 154, 'venture': 155, 'verito': 156, 'verna': 157, 'vitara': 158, 'wagon': 159, 'wrv': 160, 'x-trail': 161, 'x1': 162, 'x3': 163, 'x5': 164, 'x6': 165, 'xc60': 166, 'xc90': 167, 'xcent': 168, 'xe': 169, 'xenon': 170, 'xf': 171, 'xj': 172, 'xuv300': 173, 'xuv500': 174, 'xylo': 175, 'yeti': 176, 'zen': 177, 'zest': 178}


data = pd.read_csv('./Dataset/dataset_temp.csv') 
pickle_in = open('model.pkl', 'rb') 
regressor = pickle.load(pickle_in) 

annotation = keras.models.load_model("Image-annotation.h5")
classifier = keras.models.load_model("model_cnn.h5")

from keras.utils import load_img, img_to_array 

mapper = {1: "damaged", 0: "Not damaged"}

def predict_image(model, img): 
    # print(img.shape)
#   img = load_img(img_path, target_size=(100,100))  
#   plt.imshow(img)
    x = img_to_array(img)
#   print(np.shape(x))
    x = img.astype(np.float16)
#   print(np.shape(x))

    # img = np.reshape(img, (100, 100,3))

    x /= 255
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    answer = np.argmax(preds)
    print(mapper[answer])
    return mapper[answer]

class_names = ['Daiatsu_Core',
 'Daiatsu_Hijet',
 'Daiatsu_Mira',
 'FAW_V2',
 'FAW_XPV',
 'Honda_BRV',
 'Honda_City_aspire',
 'Honda_Grace',
 'Honda_Vezell',
 'Honda_city_1994',
 'Honda_city_2000',
 'Honda_civic_1994',
 'Honda_civic_2005',
 'Honda_civic_2007',
 'Honda_civic_2015',
 'Honda_civic_2018',
 'KIA_Sportage',
 'Suzuki_Every',
 'Suzuki_Mehran',
 'Suzuki_alto_2007',
 'Suzuki_alto_2019',
 'Suzuki_alto_japan_2010',
 'Suzuki_carry',
 'Suzuki_cultus_2018',
 'Suzuki_cultus_2019',
 'Suzuki_highroof',
 'Suzuki_kyber',
 'Suzuki_liana',
 'Suzuki_margala',
 'Suzuki_swift',
 'Suzuki_wagonR_2015',
 'Toyota HIACE 2000',
 'Toyota_Aqua',
 'Toyota_Hiace_2012',
 'Toyota_Landcruser',
 'Toyota_Passo',
 'Toyota_Prado',
 'Toyota_Vigo',
 'Toyota_Vitz',
 'Toyota_Vitz_2010',
 'Toyota_axio',
 'Toyota_corolla_2000',
 'Toyota_corolla_2007',
 'Toyota_corolla_2011',
 'Toyota_corolla_2016',
 'Toyota_fortuner',
 'Toyota_pirus',
 'Toyota_premio']

brands = list(brand_dct.keys())
def predict_img_annot(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


def predict_price(location, kms_driven, fuel_type, transmission, owner_type, mileage, engine, power, seats, used_car_price, brand, model, car_age):

    print(location)
    newbie = np.array([[location_dct[location], kms_driven, fuel_dct[fuel_type], transmission_dct[transmission], owner_dct[owner_type], mileage, engine, power, seats, used_car_price, brand_dct[brand], model_dct[model], car_age]])
    print(newbie)
    prediction=regressor.predict(newbie) 
    return prediction

def analyzer_brand(brand):
    df = pd.read_csv('./Dataset/train-new-car-dataset.csv')
    df['Car_age'] = df.Year.apply(lambda x: 2023-x)
    df_brand = df[df['Brand']==brand]
    car_no = df_brand.shape[0]
    st.subheader(f'The total no of cars of {brand} brand is {car_no}')
    avg_used_cost = df_brand['Price'].mean()
    st.subheader(f'The average cost of the {brand} Brand car is {avg_used_cost} lacs')
    cheapest_car_price = df_brand.iloc[np.argmin(df_brand['Price'])]['Price']
    cheapest_variant = df_brand.iloc[np.argmin(df_brand['Price'])][['Model','Variant']].values
    costliest_car_price = df_brand.iloc[np.argmax(df_brand['Price'])]['Price']
    costliest_variant = df_brand.iloc[np.argmax(df_brand['Price'])][['Model','Variant']].values
    st.subheader(f'The cheapest cost of the {brand} Brand car is {cheapest_car_price} lacs with the model {cheapest_variant[0]} and variant would be {cheapest_variant[1]}')
    st.subheader(f'The cheapest cost of the {brand} Brand car is {costliest_car_price} lacs with the model {costliest_variant[0]} and variant would be {costliest_variant[1]}')

    # Graphs
    # col1,col2 = st.columns(2)
    # with col1:
    st.subheader('Selling in each cities')
    plt.style.use('fivethirtyeight')
    plt.bar(df_brand['Location'], df_brand['Price'])
    plt.xticks(rotation ='vertical')
    plt.xlabel('Cities')
    plt.ylabel('Average Selling Price(in lacs)')
    st.pyplot(plt)

    # with col2:
    #     st.subheader('Selling in each cities')
    #     plt.style.use('fivethirtyeight')
    #     plt.bar(df_brand['Car_age'], df_brand['Price'])
    #     plt.xticks(rotation ='vertical')
    #     plt.xlabel('Cars Age(in years)')
    #     plt.ylabel('Average Selling Price(in lacs)')
    #     st.pyplot(plt)


        # with col2:
        #     st.header("Most busy month")
        #     busy_month = helper.month_activity_map(selected_user, df)
        #     fig, ax = plt.subplots()
        #     ax.bar(busy_month.index, busy_month.values,color='orange')
        #     plt.xticks(rotation='vertical')
        #     st.pyplot(fig)




def main():
    st.title("Car Vendor") 
    st.sidebar.title("Second Hand Car Price Prediction")
    brand=st.text_input('Brand',placeholder= 'Type Here')
    # sidebar stuff
    brandzer = st.sidebar.selectbox('Brands',brands)
    anaylzer = st.sidebar.button("Show Analysis")
    # model=st.text_input('Model', placeholder='Type Here')
    col1,col2 = st.columns(2)
    with col1:
        kms_driven=st.number_input('kms driven') 
        engine=st.number_input('Engine')
        power=st.number_input('Power')
        mileage=st.number_input('Mileage')
        used_car_price = st.number_input('Price at which car was purchased')
        # print(kms_driven,engine,power,mileage,used_car_price)
    with col2:
        model=st.selectbox('Model', list(data[data['Brand']==brand]['Model'].unique()))
        transmission = st.selectbox("Transmission",['Manual', 'Automatic'])
        owner_type = st.selectbox("Owner Type",['First', 'Second','Third','Fourth & Above'])
        location=st.selectbox('Location', ['Mumbai', 'Pune', 'Chennai', 'Coimbatore', 'Hyderabad', 'Jaipur', 'Kochi', 'Kolkata', 'Delhi', 'Bangalore', 'Ahmedabad'])
        year_of_purchase=st.selectbox('Year of purchase', [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019])
        fuel_type = st.selectbox("Fuel type",['Petrol', 'Diesel', 'CNG','LPG'])
        print(transmission,owner_type,location,year_of_purchase,fuel_type)
    seats = st.slider('Number of Seats', 2, 10)
    car_age = 2023 - year_of_purchase

    result = ""
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(image, channels="BGR")
        image1 = cv2.resize(image, (100, 100))
        image2 = cv2.resize(image, (224, 224))
        # st.image(image, channels="BGR")
        # image = Image.open(uploaded_file)
        # img = st.image(uploaded_file, caption='Uploaded image')
        # print(type(img))
        # img = st.image(uploaded_file)
        # To read file as bytes:
        # bytes_data = uploaded_file.getvalue()
        # st.write(bytes_data)
        # data = bytes_data.decode('utf-8')
        # st.text(data)
        # print(type(img))
        # imager = cv2.imread(image)
        
    print(location, kms_driven, fuel_type, transmission, owner_type, mileage, engine, power, seats, used_car_price, brand, model, car_age)
    if(st.button("Predict")): 
        result = predict_price(location, kms_driven, fuel_type, transmission, owner_type, mileage, engine, power, seats, used_car_price, brand, model, car_age)
        st.success(f"The predicted value is {result}")
        f_c = predict_image(classifier, image1)
        st.success(f"The car is {f_c}")
        f_a = predict_img_annot(annotation, image2)
        st.success(f"The car is {f_a}")
        # st.success(f"The car is {f_c}")
    st.title("Anaylsis of the car brands:")
    if(anaylzer):
        analyzer_brand(brandzer)


if __name__=='__main__': 
    main()