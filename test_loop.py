import os

def main():
    # { 0: no emotion, 1: anger, 2: disgust, 3: fear, 4: happiness, 5: sadness, 6: surprise}
    for class_label in [1, 3, 4, 5]:
        run_text = 'python3 run_pplm.py -D generic --discrim_weights "emotion.pt" --discrim_meta "emotion.json" --class_label ' + str(class_label) + ' --cond_text "John: Hey Joe" --length 50 --gamma 1.0 --num_iterations 10 --num_samples 10 --stepsize 0.04 --kl_scale 0.01 --gm_scale 0.95 --sample'
        print("class label is ",class_label)
        # run_text = 'python3 run_pplm.py -D generic --discrim_weights "emotion.pt" --discrim_meta "emotion.json" --class_label ' + str(class_label) + ' --cond_text "John: Hey Joe" --length 2 --gamma 1.0 --num_iterations 1 --num_samples 1 --stepsize 0.04 --kl_scale 0.01 --gm_scale 0.95 --sample'
        os.system(run_text)


if __name__ == '__main__':
    main()