The goal of this project is to produce a technique that can perform single-shot object recognition from a video (and do so with no pre-built filters or pre-training). So far I have acheived a basic proof-of-concept as demonstrated in the image below (the highlighting seen in this GIF represents the motion of the object that is being tracked):
![Image](./activation_map_seq.gif)

This repo contains my exploration into unsupervised learning on a single stream of video data.
There are a few different threads of ideas, but the main one, and the one that motivated me to pursue this is as follows:
- Based on what humans can do, it is possible to learn to distinguish objects based on just visual data (okay, maybe there's an "interactivity" component, but forget about that)
- So, I'm interested in looking at different ways of using neural-network based systems to learn meaningful features from the video data.
- Even more specifically, the simplest meaningful example I could think of is: it should be possible to mark a region of the first frame of the video to track, then train a neural network to predict where that region is, then as the video plays, it should be able to track the object to some degree. This is implemented in single_frame_no_patches::test_average_multi_run(). It ensembles (by averaging final activation maps) several small models to get a more consistent activation map for the object.

Another thread I've been exploring is to learn *motions* in an unsupervised way by first tracking objects, then doing unsupervised learning on the activation maps of those objects, then using sequence alignment to find time segments that correspond to each other. I've made a little bit of progress on this, e.g. comparing the correlation heatmap of unsupervised features on the same action shows some of the expected trends (slanted line-like formations). But I have a bit more to do there.
