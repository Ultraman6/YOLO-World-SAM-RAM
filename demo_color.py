import gradio as gr


# 定义回调函数，接受字典并修改其中的内容
def update_state(input_box, iso_state):
    # iso_state 是一个字典，直接修改其中的内容
    print(input_box, iso_state)
    iso_state["value"] = iso_state
    return iso_state


# 创建 Gradio 界面
with gr.Blocks() as demo:
    # 初始化一个字典作为状态
    iso_state = gr.State({})
    name = gr.Textbox(label="Enter your name")
    param = gr.Textbox(label="Enter your param")


    @gr.render(inputs=[name, param])
    def listen_name(name_v, param_v):
        iso_state.value['name'] = name_v
        iso_state.value['param'] = param_v
        print(iso_state.value)


    # 输入框，用于输入新的值
    input_box = gr.Number(label="Enter new value")

    # 输出框，用于展示更新后的字典
    output = gr.Textbox(label="Updated Dictionary")
    btn = gr.Button("Submit")
    # 当输入框的值变化时，触发 update_state 函数，并更新输出
    btn.click(fn=update_state,
              inputs={input_box, iso_state},  # 将状态和输入的值作为参数传入
              outputs=output)  # 将更新后的字典显示在输出框中

demo.launch(server_name='127.0.0.1', share=True)
