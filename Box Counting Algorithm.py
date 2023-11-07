import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('/Users/williamhartshorn/Desktop/Leaf 2.jpeg')

def make_image_square(image, output_path=None):
    # Get the dimensions of the image
    height, width, _ = image.shape
    # Determine the difference between the width and the height
    difference = abs(width - height)
    # Half the difference to be applied to the top/bottom or left/right
    pad1 = difference // 2
    pad2 = difference - pad1
    # Add white padding to make the image square
    if width > height:
        # Width is greater than height, so add padding to the top and bottom
        padded_image = cv2.copyMakeBorder(image, pad1, pad2, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    elif height > width:
        # Height is greater than width, so add padding to the left and right
        padded_image = cv2.copyMakeBorder(image, 0, 0, pad1, pad2, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    else:
        # The image is already square
        padded_image = image
    # Save the padded image if an output path is provided
    if output_path:
        cv2.imwrite(output_path, padded_image)
    return padded_image

def split_image_into_grid(image, n):
    # Ensure the image is square
    image = make_image_square(image)
    height, width, _ = image.shape
    # Calculate the size of each grid cell
    cell_size = height // (n-1) if height % (n-1) != 0 else (height // (n-1)) - 1
    # Initialize the list to hold the split images
    grid_images = []
    for row in range(n):
        for col in range(n):
            # Calculate the dimensions for the current cell
            x_start = col * cell_size
            y_start = row * cell_size
            x_end = x_start + cell_size if col != n - 1 else width
            y_end = y_start + cell_size if row != n - 1 else height

            if col == n-1:
                final_box_width = x_end-x_start
            
            # Crop the image to the current cell
            cell_image = image[y_start:y_end, x_start:x_end]
            grid_images.append(cell_image)
    return [grid_images, final_box_width, cell_size]

def get_average_color(image):
    # Calculate the mean of the image
    mean_color_per_channel = cv2.mean(image)
    
    # The result includes a mean value for each channel (B, G, R)
    # and the alpha channel if it's present. If the image does not have an alpha channel,
    # the value for alpha will be 0, and we can ignore it.
    average_color = mean_color_per_channel[:3]

    return np.mean(average_color)

def visualize_grid_stage(image, n):
    # Ensure the image is square
    image = make_image_square(image)
    height, width, _ = image.shape
    # Calculate the size of each grid cell
    cell_size = height // (n-1)
    # Initialize a copy of the image to overlay the grid and highlights
    overlay_image = image.copy()
    
    for row in range(n):
        for col in range(n):
            # Calculate the dimensions for the current cell
            x_start = col * cell_size
            y_start = row * cell_size
            x_end = x_start + cell_size if col != n - 1 else width
            y_end = y_start + cell_size if row != n - 1 else height
            
            # Crop the image to the current cell
            cell_image = image[y_start:y_end, x_start:x_end]
            
            # Check if the cell contains both black and white pixels
            if np.any(cell_image != 0) and np.any(cell_image != 255):
                # Highlight the square if it contains both black and white pixels
                cv2.rectangle(overlay_image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)  # Red rectangle
            else:
                # Draw the grid lines
                cv2.rectangle(overlay_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)  # Green rectangle
    
    # Display the image with the grid and highlighted squares
    cv2.imshow('Grid Stage Visualization', overlay_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def box_count(image, m=20):
    image = make_image_square(image)
    log_N = []
    log_E = []
    stage = 1
    for n in range(5, int(image.shape[0]/2)+1): 
        n = int(n)
        if split_image_into_grid(image, n)[2] >= split_image_into_grid(image, n)[1]:
            print('Boxes: ',n,' Normal Size: ', split_image_into_grid(image, n)[2],' Final Size: ', split_image_into_grid(image, n)[1])  
            count = 0
            boxes = split_image_into_grid(image, n)[0]
            
            for box in boxes:
                if np.any(box != 0) and np.any(box != 255):
                    count += 1
            
            log_N.append(np.log(count))
            log_E.append(np.log(1/n))
            
            # Visualize the grid and highlighted squares for a specific stage
            # Change this value to visualize a different stage
            # visualize_grid_stage(image, n)
            stage +=1
    
    m, c = np.polyfit(log_E, log_N, 1)
    plt.scatter(log_E, log_N)
    plt.plot(log_E, [m*x+c for x in log_E])
    plt.xlabel('Log(ε)')
    plt.ylabel('Log(N)')
    plt.title(f'Slope (fractal dimension): {m}')
    plt.show()

box_count(image)
