import numpy as np

def render(width=800, height=450):
    aspect = width / height
    origin = np.array([0,0,0])
    sphere_center = np.array([0,0,-1])
    radius = 0.5

    img = np.zeros((height, width, 3), dtype=np.float32)
    for j in range(height):
        for i in range(width):
            u = (i + 0.5) / width * 2 - 1
            v = (j + 0.5) / height * 2 - 1
            dir = np.array([u * aspect, -v, -1])
            dir /= np.linalg.norm(dir)
            # ray-sphere intersection
            oc = origin - sphere_center
            b = np.dot(dir, oc)
            c = np.dot(oc, oc) - radius**2
            disc = b*b - c
            if disc >= 0:
                img[j,i] = [1,0,0]
    return (img*255).astype(np.uint8)


if __name__ == "__main__":
    from PIL import Image
    img = render()
    Image.fromarray(img).save("sphere_cpu.png")
