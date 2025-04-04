import time

from openai import OpenAI
from new_wzry_ai.config.default_config import openai_api_key_4_0_2
from new_wzry_ai.config.prompt import position_predict_prompt
from new_wzry_ai.utils.other_tools import image_to_base64
from new_wzry_ai.utils.screen import ScreenCapture
from new_wzry_ai.config.default_config import TemplateRegion
from new_wzry_ai.utils.heros_position_detector import hero_position_detector
from new_wzry_ai.config.default_config import other_status

class ChatGptTool:
    # 类初始化
    __instance=None


    def __init__(self):
        self.__client = OpenAI(
            # This is the default and can be omitted
            api_key=openai_api_key_4_0_2
        )
        self.__message: list[dict] = position_predict_prompt

    def get_instance(self):
        if self.__instance is None:
            __instance=ChatGptTool()
        return __instance

    #重置message
    def clear_message(self):
        self.__message=position_predict_prompt

    # 进行对话
    def chat_with_gpt_with_text(self,usercontent) -> str | None:
        usermessage={"role":"user","content":usercontent}
        self.__message.append(usermessage)
        completion = self.__client.chat.completions.create(
            # model="gpt-4o",
            model="gpt-4o-2024-08-06",
            messages=self.__message,
        )
        aicontent = completion.choices[0].message.content
        self.__message.append({"role":"assistant","content":aicontent})
        return aicontent

    # 进行对话
    def chat_with_gpt(self) -> str | None:
        completion = self.__client.chat.completions.create(
            # model="gpt-4o",
            model="gpt-4o-2024-08-06",
            messages=self.__message,
        )
        aicontent = completion.choices[0].message.content
        self.__message.append({"role": "assistant", "content": aicontent})
        return aicontent

    def append_user_message(self,content:str):
          self.__message.append({"role":"user","content":content})


    def append_user_message_with_image(self,text:str,image):
        base64_image = image_to_base64(image)
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text

                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"  # 使用Base64编码的图片
                        }
                    }
                ]
            }
        ]
        self.__message.extend(message)

    def chat_with_gpt_with_image(self):
        image_path = 'image/1.png'
        base64_image = image_to_base64(image_path)
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "描述这张图片,并给出黄色星星和红色方块的坐标，假设该图片坐标中心为左上角，并解析图中的文字内容"

                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"  # 使用Base64编码的图片
                        }
                    }
                ]
            }
        ]
        self.__message.extend(message)
        completion = self.__client.chat.completions.create(
            # model="gpt-4o",
            model="gpt-4o-2024-08-06",
            messages=self.__message,
        )
        aicontent = completion.choices[0].message.content
        self.__message.append({"role": "assistant", "content": aicontent})
        return aicontent

chatgpt_tool  = ChatGptTool()

#通过大语言模型更新目标位置
def update_movetarget():
        screen_capture= ScreenCapture()
        x1, y1, x2, y2 = TemplateRegion.CHARACTER_AREA
        while True:
            try:
                image=screen_capture.capture()
                image=image[y1:y2, x1:x2]
                my_position, teammates_temp,enemies_temp = hero_position_detector.get_hero_position(image)
                teammates=[[int(num) for num in sublist] for sublist in teammates_temp]
                enemies=[[int(num) for num in sublist] for sublist in enemies_temp]
                #print(f"teammate:{teammates}    enemy:{enemies}")
                if my_position is None:
                    continue
                #print(my_position)
                my_x, my_y = int(my_position[0]), int(my_position[1])
                #print(total_time)
                #print(f"当前自己所在位置坐标({my_x},{my_y})")
                """
                chatgpt_tool.append_user_message_with_image(f"当前对局时间：{int(other_status['time'])}秒，当前自己所在位置坐标({my_x},{my_y}),队友英雄位置：{teammates},敌方英雄位置：{enemies}", image)
                ai_content = chatgpt_tool.chat_with_gpt()
                print(ai_content)
                ai_reply = eval(ai_content)
                x,y=ai_reply["position"][0]+TemplateRegion.CHARACTER_AREA[0],ai_reply["position"][1]+TemplateRegion.CHARACTER_AREA[1]
                other_status["move_target"]=(x,y)
                other_status["move_reason"]=ai_reply["reason"]
                """

                #print(f"x:{x1+x},y:{y1+y}")
                time.sleep(30)
            except Exception as e:
                print("====================== error =======================")
                print(e)
                print("====================== error =======================")

