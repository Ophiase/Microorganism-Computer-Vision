from load import load_video
from optical_flow import compute_optical_flow, preview_optical_flow, visualize_optical_flow_arrows

###################################################################################


VIDEO = "data/342430.avi"

###################################################################################


def main() -> None:
    video = load_video(VIDEO, interval=(0, 10), verbose=True)
    video_optical_flow = compute_optical_flow(video)

    print(video.shape)
    print(video_optical_flow.shape)

    # preview_optical_flow(video, video_optical_flow)
    # visualize_optical_flow_arrows(video, video_optical_flow)


if __name__ == "__main__":
    main()
