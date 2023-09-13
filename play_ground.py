import numpy as np
import matplotlib.pyplot as plt

# Generate some fake EMD transport map for demonstration
# Assume images are of size 10x10
transport_map = np.random.rand(10, 10, 10, 10)
transport_map /= transport_map.sum((2, 3), keepdims=True)  # Normalize so that sum along each outgoing pixel is 1

# Fake image1 and image2
image1 = np.random.rand(10, 10)
image2 = np.random.rand(10, 10)


def plot_transport(x, y):
    plt.clf()  # Clear the current figure
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title('Image 1')
    plt.imshow(image1, cmap='gray')
    plt.scatter([y], [x], c='red')

    plt.subplot(1, 3, 2)
    plt.title('Transport Map')
    plt.imshow(transport_map[int(x), int(y), :, :], cmap='viridis')

    plt.subplot(1, 3, 3)
    plt.title('Image 2')
    plt.imshow(image2, cmap='gray')
    plt.imshow(transport_map[int(x), int(y), :, :], cmap='viridis', alpha=0.5)

    plt.show()


def on_click(event):
    x, y = event.ydata, event.xdata
    if event.inaxes is None or x is None or y is None:
        return

    # Round to the nearest integer coordinate
    x, y = round(x), round(y)

    # Check bounds
    if 0 <= x < 10 and 0 <= y < 10:
        plot_transport(x, y)


# Initial plot
plot_transport(0, 0)

# Connect the click event to the plot update function
cid = plt.gcf().canvas.mpl_connect('button_press_event', on_click)

# Keep the plot window open
plt.show()
