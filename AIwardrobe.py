import requests
import json
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from typing import List
import warnings
from PIL import Image, ImageFile


def get_weather_description(location: str, user: str) -> tuple:
    url = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php"
    data = "rhrread"
    lang = "tc"
    filepath = f"Users\{user}"
    file_names = os.listdir(filepath)
    response = requests.get(f"{url}?dataType={data}&lang={lang}")
    weather_data = response.json()

    temp = None
    for temp_record in weather_data['temperature']['data']:
        if temp_record['place'] == location:
            temp = temp_record['value']
            break

    humidity = None
    for humidity_record in weather_data['humidity']['data']:
        if humidity_record['place'] == '香港天文台':
            humidity = humidity_record['value']
            break

    rainfall = None
    for rainfall_record in weather_data['rainfall']['data']:
        if rainfall_record['place'] == location:
            rainfall = rainfall_record['max']
            break

    uv_index = None
    for uv_record in weather_data['uvindex']:
        if uv_record['place'] == '京士柏':
            uv_index = uv_record['value']
            break

    print(f"Temp: {temp}°C   Humid: {humidity}%   Rainfall: {rainfall} mm   UV_index: {uv_index}\n")
    weather_description = [f"{temp}",f"{humidity}",f"{rainfall}",f"{uv_index}"]

    return weather_description,filepath


def get_outfit_advice(weather_description: List[str],filepath: str,specialneed: str) -> str:
    file_names = os.listdir(filepath)
    available_outfits = [f for f in file_names if os.path.isfile(os.path.join(filepath, f))]

    weather_description = (
        f"当前天气条件如下:\n"
        f"温度: {weather_description[0]}°C\n"
        f"湿度: {weather_description[1]}%\n"
        f"降雨量: {weather_description[2]} mm\n"
        f"紫外线指数: {weather_description[3]}\n\n"
        f"可选的穿搭组合有:\n"
        f"{', '.join(available_outfits)}\n\n"
        f"根据上述天气数据和可选的穿搭组合，请推荐最适合的一套衣服。"
        f"特殊需求：{specialneed}"
        f"请只回复你推荐的穿搭图片的文件名，包含一句最简短的原因，以英文回答。格式为xxx.jpg:原因"
    )

    url = "https://api.deepseek.com/chat/completions"

    payload = json.dumps({
        "messages": [
            {
                "content": "你是一个专业的穿搭助手，需要根据天气情况从给定的穿搭选项中选择最合适的一套。",
                "role": "system"
            },
            {
                "content": weather_description,
                "role": "user"
            }
        ],
        "model": "deepseek-chat",
        "temperature": 0.3,  # 降低随机性，使选择更确定性
        "max_tokens": 50
    })

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': 'Bearer sk-8eb9823ca5184eb7b675d4b41131cb49'
    }

    response = requests.post(url, headers=headers, data=payload)
    response_data = response.json()

    # 获取AI推荐的穿搭
    ai_recommendation = response_data["choices"][0]["message"]["content"].strip()


    return ai_recommendation

def show_outfit_image(recommended_outfit: str,filepath: str):
    recommended_outfit = recommended_outfit.split(': ')
    print(f"Based on the current weather, Deepseek recommends the following outfits: {recommended_outfit[0]}")
    print(f"Reason: {recommended_outfit[1]}")
    outfit_path = os.path.join(filepath, recommended_outfit[0])
    #outfit_path = os.path.join(filepath, "Polo&Pants.jpg")
    try:
        img = Image.open(outfit_path)
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')  # 不显示坐标轴
        plt.title("Recommended Outfit: " + os.path.basename(outfit_path).split('.')[0])
        plt.show()
    except Exception as e:
        print(f"无法打开图片: {e}")


def main():
    location = "沙田"  # Shatin

    user = "Victor" # use face recognition to decide user name

    weather_description,filepath = get_weather_description(location,user) #get weather data
    #weather_description = ["20","90","20","78"]
    recommended_outfit = get_outfit_advice(weather_description,filepath,specialneed= "I need to attend disco party") #access deepseek v3 and get recommendation
    show_outfit_image(recommended_outfit,filepath) #display corresponding image


if __name__ == "__main__":
    main()