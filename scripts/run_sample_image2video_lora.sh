#!/bin/bash

python3 sample_image2video.py \
   --prompt "Two people hugged tightly, In the video, two people are standing apart from each other. They then move closer to each other and begin to hug tightly. The hug is very affectionate, with the two people holding each other tightly and looking into each other's eyes. The interaction is very emotional and heartwarming, with the two people expressing their love and affection for each other." \
   --i2v-image-path ./assets/demo/i2v_lora/imgs/embrace.png \
   --lora-path ./ckpts/hunyuan-video-i2v-720p/lora/embrace_kohaya_weights.safetensors \
   --model HYVideo-T/2 \
   --i2v-mode \
   --i2v-resolution 720p \
   --i2v-stability \
   --infer-steps 50 \
   --video-length 129 \
   --flow-reverse \
   --flow-shift 5.0 \
   --embedded-cfg-scale 6.0 \
   --seed 0 \
   --use-cpu-offload \
   --save-path ./results \
   --use-lora \
   --lora-scale 1.0 \

# More examples
#    --prompt "rapid_hair_growth, The hair of the characters in the video is growing rapidly. The character's hair undergoes a dramatic transformation, growing rapidly from a short, straight style to a long, wavy one. Initially, the hair is a light blonde color, but as it grows, it becomes darker and more voluminous. The character's facial features remain consistent throughout the transformation, with a slight change in the shape of the jawline as the hair grows. The clothing changes from a simple, casual outfit to a more elaborate, fashionable ensemble that complements the longer hair. The overall appearance shifts from a casual, everyday look to a more stylish, sophisticated one. The character's expression remains calm and composed throughout the transformation, with a slight smile as the hair grows." \
#    --i2v-image-path ./assets/demo/i2v_lora/imgs/hair_growth.png \
#    --lora-path ./ckpts/hunyuan-video-i2v-720p/lora/hair_growth_kohaya_weights.safetensors \
