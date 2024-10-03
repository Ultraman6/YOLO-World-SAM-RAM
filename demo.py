import gradio as gr

# 定义嵌套字典
choices = {
    '水果': {
        '柑橘': {
            '橙子': 1,
            '柠檬': 2
        },
        '浆果': {
            '草莓': 3,
            '蓝莓': 4
        }
    },
    '蔬菜': {
        '叶菜': {
            '菠菜': 5,
            '生菜': 6
        },
        '根菜': {
            '胡萝卜': 7,
            '甜菜': 8
        }
    },
    '谷物': {
        '稻米': 9,
        '小麦': 10
    }
}

# 定义最大深度
MAX_DEPTH = 10


def get_choices_at_path(choices_dict, path):
    current = choices_dict
    for key in path:
        if key in current:
            current = current[key]
        else:
            return []
    if isinstance(current, dict):
        return list(current.keys())
    else:
        return []


def get_final_value(choices_dict, path):
    current = choices_dict
    for key in path:
        if key in current:
            current = current[key]
        else:
            return "无效的选择路径"
    if isinstance(current, dict):
        return "请选择更深一级的选项"
    else:
        return current


def on_select(depth, selected, path):
    new_path = path[:depth]  # 保留当前深度之前的选择
    if selected:
        new_path.append(selected)

    next_choices = get_choices_at_path(choices, new_path)
    final_value = get_final_value(choices, new_path)

    updates = [new_path]

    if depth < MAX_DEPTH:
        if next_choices:
            updates.append(gr.Dropdown.update(choices=next_choices, visible=True, value=None))
        else:
            # 隐藏后续所有下拉列表
            for _ in range(depth + 1, MAX_DEPTH):
                updates.append(gr.Dropdown.update(choices=[], visible=False, value=None))

    updates.append(final_value)

    return updates


with gr.Blocks() as demo:
    gr.Markdown("### 动态递归多级下拉列表示例（支持任意深度）")

    # 初始化状态，存储用户的选择路径
    path_state = gr.State([])

    # 创建多个下拉列表
    dropdowns = []
    for i in range(MAX_DEPTH):
        dropdown = gr.Dropdown(
            label=f"选择第 {i + 1} 级",
            choices=[],
            visible=False
        )
        dropdowns.append(dropdown)

    # 文本框显示最终的数值
    output = gr.Textbox(label="对应的数值", interactive=False)

    # 显示第一级下拉列表
    dropdowns[0].update(choices=list(choices.keys()), visible=True)

    # 为每个下拉列表绑定事件
    for i in range(MAX_DEPTH):
        dropdowns[i].change(
            fn=on_select,
            inputs=[gr.Number(value=i), dropdowns[i], path_state],
            outputs=[path_state] + dropdowns[i + 1:MAX_DEPTH] + [output]
        )

    # 布局下拉列表
    for dropdown in dropdowns:
        demo.append(dropdown)

    # 添加输出文本框
    demo.append(output)

demo.launch(server_name='127.0.0.1', share=True)
