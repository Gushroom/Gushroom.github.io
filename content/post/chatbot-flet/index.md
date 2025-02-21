---
title: Building a Chatbot with Flet
description: Building my own ChatGPT clone with Flet
date: 2025-02-20 00:00:00+0000
tags:
    - llm
    - flet
    - app
---

# Building a Chatbot with Flet
ChatGPT has becoming more and more difficult to use with VPN. So I decided to build one with API. Getting famailar with Flet(flutter) is an extra because I want to build more stuff on this platform.

## Flet
The interaction logic is very easy: `onclick=function()`. With buttons:
```python
send_button = ft.ElevatedButton(
	text="Send", 
	on_click=send_message, 
	width=page.width * 0.2,
	style=ft.ButtonStyle(
		color=ft.Colors.WHITE,
		bgcolor=ft.Colors.BLUE_400,
	)
)
```
To enable markdown in response, we need to replace the `TextField` with a `Markdown` element.
```python
output_box = ft.Markdown(
	value="",
	selectable=True,
	extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
	on_tap_link=lambda e: page.launch_url(e.data),
	width=page.width * 0.7,
)
```

Saving chat history and memory to local storage: `page.client_storage.set(key, value)`

## LLM
A basic template to use LLM APIs:
```python
class ChatBot():
    def __init__(self, api_key, base_url=None, system_prompt=None, model="o3-mini-high"):
        default_system_prompt = "You are a helpful assistant."
        default_base_url = "https://api.openai-next.com/v1"
        self.system_prompt = system_prompt if system_prompt else default_system_prompt 
        self.messages=[{"role": "system", "content": self.system_prompt}]
        self.memories=[]
        self.model = model
        self.base_url = base_url if base_url else default_base_url
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

    def chat(self, user_message):
        self.messages.append({"role": "user", "content": user_message})
        completion = self.client.chat.completions.create(
            model=self.model,
            stream=False,
            messages=self.messages
        )
        response = completion.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response})
        return response
```

