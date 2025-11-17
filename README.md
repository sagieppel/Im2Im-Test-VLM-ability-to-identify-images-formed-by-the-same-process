# Im2Im: Test vision language models (VLM)  ability to identify and match different images that were formed by the same system/process/model

Given a set of reference images, all created by the same process/model/simulation, and a set of test images created  by different processes, the VLM is asked to identify which set of images was created by the same process as the reference images.

The code runs on the [SciTextures dataset.](https://zenodo.org/records/17485502)

For more details on the method, see: [SciTextures: Collecting and Connecting Visual Patterns, Models, and Code Across Science and Art.](https://arxiv.org/pdf/2511.01817) 

​![](/im2im.jpg)

# How to use:​

## 1. Download and extract the [SciTextures dataset from here](https://zenodo.org/records/17485502/files/Scitexture_Full_110K_images_jpg_format.zip?download=1)​

## 2. Update your preferred API key at API_KEYS.py.  ​

​The code relies on API for major AI LVLM suppliers (GPT, Gemini, Qwen, DeepSeek, Claude, Grok...). You need API key (and purchase tokens) for at least one of those (GPT-5 is recommended). 

The code supports APIs from OpenAI, Together.AI (Qwen, LLama), Grok,Gemini, and Claude. You can add other APIs to VisualQuestion.py.

​

## 3. Run test (Run_Test_im2im.py)

In **Run_Test_im2im.py** in **__main__** set path to the  SciTextures images main folder.



**images_main_dir = r"Scitexture/images/"** ​



**out_dir = r"output_Result_dir//"**

​And run the script, the results will be displayed on screen and saved in ***out_dir***.
