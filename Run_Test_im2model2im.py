import os
import random
import math
import cv2
import os
import random
import VisualQuestion as VQ
import json_pkl
'''
Im2Model2Im: Test vision language models ability to identify and match difference images that were formed by the same model
The VLM is first asked to infer the model/proccess that formed each image, the textual description of the model infered from each image. 
is matched to the textual description of the model infer from the referance images.
Given a set of reference images all created by the same proccess/model/simulation and a set of test images created 
by different simulation the VLM is asked to identify which set of images were created by the same proccess as the referance images.
The code run on the SciTextures datasett and need API key for the specific model used (set in API_KEY.py)
'''

##########################################################################################################################

def _collect_groups(images_main_dir): #
    """Map: group_name -> list of absolute image paths in group_dir/textures/"""
    im_data = {}
    for dr in sorted(os.listdir(images_main_dir)):
        img_dir = os.path.join(images_main_dir, dr)
        if not os.path.isdir(img_dir):
            continue
        images = [
            os.path.join(img_dir, f)
            for f in sorted(os.listdir(img_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if images:
            im_data[dr] = images
    return im_data



#############################image to image matching test###############################################################################3

def run_test_im2im(
    images_main_dir,  # images main dir (Scitexture/images/
    max_questions: int, # max questions per test
    num_reference: int, # Number referance images (from the same model)
    num_neg_sample: int,# number of test images not belonging to the model
    model: str = "gpt-5", # VLM use for the test
    retries: int = 4, # Retries if model fail to give answer
    display: bool = False, # For debug
    outdir: str = "out_dir//",  # output dir
    unify_images: bool = True, # display question as single dir

):
    """

    """


    im_data = _collect_groups(images_main_dir)
    groups = list(im_data.keys())
    if not groups:
        raise RuntimeError(f"No groups with images found under: {main_dir}")

    results = {"correct": 0, "wrong": 0, "fail": 0}


    n_done = 0

    for ref_group_name in groups:
            if n_done >= max_questions:
                break

            ref_group = im_data[ref_group_name]
            if len(ref_group) < num_reference + 1:
                print(f"Skipping '{ref_group_name}': not enough images ({len(ref_group)} found).")
                continue

            # pick refs + positive
            picked = random.sample(ref_group, num_reference + 1)
            pos_choice_path = picked[0]
            ref_labels = [f"reference {i+1}" for i in range(num_reference)]
          #  test_images = {ref_labels[i]: picked[i + 1] for i in range(num_reference)}
            test_images={} # choices
            ref_images = {ref_labels[i]: picked[i + 1] for i in range(num_reference)} # Reference images
            # negatives
            other_groups = [g for g in groups if g != ref_group_name]
            # if len(other_groups) < num_neg_sample:
            #     print(f"Skipping '{ref_group_name}': need {num_neg_sample} negative groups, "
            #           f"have {len(other_groups)}.")
            #     continue
            neg_groups = random.sample(other_groups, num_neg_sample)

            # build test options (N negatives + 1 positive) in random position
            total_tests = num_neg_sample + 1
            labels = [f"test{i+1}" for i in range(total_tests)]
            correct_pos = random.randint(0, total_tests - 1)
            correct_choice = labels[correct_pos]
            options = labels[:]

            # assign images to test labels
            ng_idx = 0
            for k, lab in enumerate(labels):
                if k == correct_pos:
                    test_images[lab] = pos_choice_path
                else:
                    # pick one image from the ng_idx-th negative group
                    img = random.choice(im_data[neg_groups[ng_idx]])
                    test_images[lab] = img
                    ng_idx += 1
            #--------------Unify images ----------------------------------------------
            if unify_images:
                test_im_path=""
                #if len(outdir)>0:
                if not os.path.exists(outdir): os.mkdir(outdir)
                ref_im_path = outdir+"//"+str(n_done)+"_ref.jpg"
                test_im_path = outdir + "//" + str(n_done) + "_test.jpg"
                json_pkl.save_json(test_images,outdir+"//"+str(n_done)+".json")
                tim , test_images = VQ.unify_image(data=test_images,num_columns=4,labels=["test"],out_file=test_im_path,disp=False)
                rim, ref_images = VQ.unify_image(data=ref_images, num_columns=4, labels=["reference"],out_file=ref_im_path, disp=False)

            # ---------- ask the model ----------
            q_txt_model_ref = ("\n You are given set of images containing textures/pattern that was all created by the same model/algorithm/simulation."
                           "\n I want you to try and infer the model/mechanism/algorithm that form this images."  # (for example monte carlo/random walk/game of life/bolzman lattice)(..."
                           "\n Your answer should come as JSON dictionary of the following format:\n"
                           "{"
                           "\n'model':your best estimate as to the model that form this pattern (dont explain how you got it just the model)."
                            "\n'explain': explanation of how you got to it."
                           "\n}")

            q_txt_model_test = ("\n You are given set of images divided into several group named: " + str(options) +
                           "\n Each images in the same group were generated by the same model/algorithm/simulation,"
                           "\n For each group of images I want you to try and infer the model/mechanism/algorithm that form it."  # (for example monte carlo/random walk/game of life/bolzman lattice)(..."
                           "\n Your answer should come as JSON dictionary of the following format:\n"
                           "{\n'" + options[
                               0] + "':{'model':your best estimate as to the model that form this pattern (dont explain how you got it just the model)."
                                    "'explain': explanation of how you got to to it.}"
                                    "\n'" + options[1] + "': .....}")



            outcome = "fail"
            for _ in range(retries):
                models_desc_ref = VQ.get_response_image_txt_json(text=q_txt_model_ref, img_path=ref_images, model=model)
                models_desc_test = VQ.get_response_image_txt_json(text=q_txt_model_test, img_path=test_images, model=model)

                model_txt_test = ""
                for ky in models_desc_test:
                    if 'model' in models_desc_test[ky]:
                        model_txt_test += "\n" + ky + ":  " + models_desc_test[ky]['model']
                    else:
                        print("Missing model in ", ky, "\n", models_desc_test[ky])
                if not 'model' in models_desc_ref:
                    print("No reference model ")
                    continue

                qtxt = str(
                    "Here are description of several groups of algorithms/methods/models which might be inaccurate or contain errors."
                    "***" + model_txt_test + "***"
                                        "\n You are also given a description reference model:"
                                                                                      "***\n" + models_desc_ref['model'] + "\n ***"
                                   "\nWhich  of the models " + str(
                        options) + " is most similar to the reference model"
                    "\nProvide your answer as json dictionary of the following structure"
                    "\n{'answer': <the best suited model must be one of:" + str(
                        options) + ">, 'explanation':<explain your answer>}")

                print(qtxt)
                ans = VQ.get_response_image_txt_json(text=qtxt, model=model)
                print(ans)
                if 'answer' in ans and ans['answer'] in options:
                    if ans['answer'] == correct_choice:
                        outcome = "correct"
                    else:
                        outcome = "wrong"
                    break

            print(f"[{ref_group_name}] -> {outcome}")
            results[outcome] += 1
            n_done += 1
            print(results)
            #***************************Display and save **********************************************************
            #================== save============
            if len(error_dir)>0: # and  (outcome=="wrong" or outcome=="fail"):
                if not os.path.exists(error_dir): os.mkdir(error_dir)
                spec_er_dir=os.path.join(error_dir,ref_group_name+"_"+outcome)
                if not os.path.exists(spec_er_dir): os.mkdir(spec_er_dir)

                txt = q_txt_model_ref + "\n\nAnswer:\n\n" + str(models_desc_ref)+ q_txt_model_test + "\n\nAnswer:\n\n" + str(models_desc_test)+ "\n" + qtxt + "\n\nAnswer:\n\n" + str(ans)
                with open(spec_er_dir + "/QA.txt", "w") as fl:
                    fl.write(txt)
                json_pkl.save_json({"q1": q_txt_model_ref, "a1": models_desc_ref, "q2": q_txt_model_ref, "a2": models_desc_ref, "q3": qtxt, "a3": ans}, spec_er_dir + "/QA.json")
                json_pkl.save_json({"q1": q_txt_model_ref, "a1": models_desc_ref, "q2": q_txt_model_ref, "a2": models_desc_ref,"q3": qtxt, "a3": ans}, spec_er_dir + "/QA.pkl")




                for ky in test_images:
                    if ky == correct_choice:
                        cv2.imwrite(spec_er_dir + "//"+ky + "correct.jpg", cv2.imread(test_images[ky]))
                    elif ("answer" in ans) and ky== ans["answer"]:
                        cv2.imwrite(spec_er_dir + "//" + ky + "choice.jpg", cv2.imread(test_images[ky]))
                    else:
                        cv2.imwrite(spec_er_dir + "//" +ky+".jpg", cv2.imread(test_images[ky]))
                with open(spec_er_dir+"//data.txt","w") as fl: fl.write(ref_group_name+"\n"+str(ans)+"\ncorrect"+correct_choice)
    # display
            if display:
                for ky in test_images:
                    if ky == correct_choice:
                        cv2.imshow(ky + "correct", cv2.imread(test_images[ky]))
                    else:
                        cv2.imshow(ky + "  " + ref_group_name, cv2.imread(test_images[ky]))
                        print(ref_group_name)
                cv2.waitKey()
            #=*******************************************************



    total = results["correct"] + results["wrong"] + results["fail"]
    results["total"] = total
    results["accuracy"] = (results["correct"] / total) if total else 0.0
    return results

########################## Run test with multiple models #####################################################################################################33
# def run_test_multi_model(
#         main_in_dir,
#         main_outdir,
#         max_questions=100,
#         num_reference=3,
#         num_neg_sample=10,
#         single_img=True,
#         skip_exist=True
# ):
#     if not os.path.exists(main_outdir):
#         os.mkdir(main_outdir)
#     openai_models = ["gpt-5-mini", "gpt-5"]  # , "gpt-oss-120b", "gpt-oss-20b"]
#     together_models = ["google/gemma-3n-E4B-it", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
#                        "meta-llama/Llama-4-Scout-17B-16E-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct"]
#     gemini_models = ["gemini-2.5-pro", "gemini-2.5-flash"]
#     grok_models = ["grok-4-fast-reasoning", "grok-4-fast-non-reasoning", "grok-4"]
#     claude_models = ["claude-sonnet-4-5-20250929"]
#     combine_list = openai_models + gemini_models + together_models + gemini_models + grok_models + claude_models
#
#     for model in combine_list:
#         model_simple_name = model.replace(".", "").replace(" ", "").replace("-", "_").replace("/", "_").replace(r"\\",
#                                                                                                                 r"_")
#
#         if os.path.exists( main_outdir + "//" + model_simple_name + ".pkl") and skip_exist: continue
#         model_simple_name = model.replace(".", "").replace(" ", "").replace("-", "_").replace("/", "_").replace(r"\\",
#                                                                                                                 r"_")
#         stats = run_test_im2im(
#             main_dir=main_in_dir,
#             max_questions=max_questions,
#             num_reference=num_reference,
#             num_neg_sample=num_neg_sample,
#             model=model,
#             # "grok-2-vision",#"Qwen/Qwen2.5-VL-72B-Instruct",#"gpt-5-mini",#"claude-3-5-sonnet-latest" ,#"grok-2-vision",#"gemini-2.5-flash", #"Qwen/Qwen2.5-VL-72B-Instruct", #"meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",#"gpt-5-mini",
#             retries=2,
#             display=False,  # show fullscreen combined window for each question
#             error_dir=main_outdir+"//"+model_simple_name+"_results//",
#             unify_images=single_img,
#             outdir=main_outdir+"//"+model_simple_name+"//"
#         )
#         json_pkl.save_json(stats,main_outdir+"//"+model_simple_name+".json")
#         json_pkl.save_pkl(stats, main_outdir + "//" + model_simple_name + ".pkl")
#
#
#         print(main_outdir+"//"+model_simple_name)



############################################main ###########################################################################################
# ----------------------------- __main__ -----------------------------

############################################main ###########################################################################################
# ----------------------------- __main__ -----------------------------

if __name__ == "__main__":
    images_main_dir =  r"Scitexture/images/"  # Image main dir from the SciTextures dataset
    out_dir = r"output_dir//"  # Output dir where results will be saved
    model = "gpt-5"


    run_test_im2im(
            images_main_dir,
            max_questions=100,  # max questions per test
            num_reference=3,  # Number referance images (from the same model)
            num_neg_sample=10,  # number of test images not belonging to the model
            model=model,  # VLM use for the test
            outdir=out_dir,  # output dir
    )
