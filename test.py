from load import load_video
from optical_flow import (
    compute_optical_flow, visualize_optical_flow_with_arrows,
    visualize_optical_flow_with_slider, visualize_video_and_optical_flow_side_by_side
)

###################################################################################


VIDEO = "data/342430.avi"

###################################################################################


def main() -> None:
    video = load_video(VIDEO, interval=(0, 10), verbose=True)
    video_optical_flow = compute_optical_flow(video)

    print(video.shape)
    print(video_optical_flow.shape)

    #########################################

    print("Visualize")

    visualize_video_and_optical_flow_side_by_side(
        video, video_optical_flow).show()

    # preview_optical_flow(video, video_optical_flow)
    # visualize_optical_flow_single_frame(video, video_optical_flow).show()
    # visualize_optical_flow_with_slider(video, video_optical_flow).show()


if __name__ == "__main__":
    main()
