from flask import request,Blueprint
from new_wzry_ai.core.memory import memory



def print_structure(data, indent=0):
    # 打印当前数据的类型
    print("  " * indent + f"Type: {type(data)}")

    # 如果数据是列表，递归处理其中的元素
    if isinstance(data, list):
        for item in data:
            print_structure(item, indent + 1)


bp = Blueprint("transport", __name__, url_prefix="/transport")


@bp.route('/receive_experiences', methods=['POST'])
def upload_experiences():
    try:
        """接收客户端发送的经验数据"""
        experience_batch = request.get_json()
        #print(f"Received experience batch: {len(experience_batch)}")
        #print_structure(experience_batch)
        #print(experience_batch[0].keys())
        for experience in experience_batch:
            memory.push(experience[0],
                        experience[1],
                        experience[2],
                        experience[3],
                        experience[4])
        print("OK")
        return "OK",200
    except Exception as e:
        print("====================== error =======================")
        print(e)
        print("====================== error =======================")
        return "Error occurred while processing the request", 500

