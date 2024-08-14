package Sprite;


import javafx.scene.Node;
import javafx.scene.image.ImageView;
import javafx.scene.transform.Rotate;
import javafx.scene.transform.Transform;

import java.util.List;

import Game.Renderer;

/**
 * Class for representing sprite.
 */
public class StaticSprite
{
    private double x, y, z;
    private double height, width;
    private final ImageView img;

    /**
     * Sets sprites x coordinate.
     * @param x new x coordinate.
     */
    public void setX(double x)
    {
        this.x = x;
        img.translateXProperty().setValue(x - (width / 2d));
    }

    /**
     * Sets sprites y coordinate.
     * @param y new y coordinate.
     */
    public void setY(double y)
    {
        this.y = y;
        img.translateYProperty().setValue(y - (height));
    }

    /**
     * Sets sprites z coordinate.
     * @param z new z coordinate.
     */
    public void setZ(double z)
    {
        this.z = z;
        img.translateZProperty().setValue(z);
    }

    /**
     * Sets height of sprite and centers it.
     * @param height
     */
    public void setHeight(double height)
    {
        this.height = height;
        img.setFitHeight(height);
        img.translateYProperty().setValue(y - (height));
    }

    /**
     * Sets width of sprite and centers it.
     * @param width
     */
    public void setWidth(double width)
    {
        this.width = width;
        img.setFitWidth(width);
        img.translateXProperty().setValue(x - (width / 2d));
    }

    /**
     * @return x coordinate of sprite.
     */
    public double getX() { return x; }

    /**
     * @return y coordinate of sprite.
     */
    public double getY() { return y; }

    /**
     * @return z coordinate of sprite.
     */
    public double getZ() { return z; }

    /**
     * @return height of sprite.
     */
    public double getHeight() { return height; }

    /**
     * @return width of sprite.
     */
    public double getWidth() { return width; }

    /**
     * @return ImageView of sprite.
     */
    public ImageView getImage() { return img; }


    /**
     * Creates sprite anchored to center of bottom edge.
     * @param x x coordinate of sprite.
     * @param y y coordinate of sprite.
     * @param z z coordinate of sprite.
     * @param file string specifying the path of sprite.
     */
    public StaticSprite(double x, double y, double z, String file)
    {
        img = new ImageView(file);

        height = img.getImage().getHeight();
        width = img.getImage().getWidth();

        setX(x);
        setY(y);
        setZ(z);
    }

    /**
     * Creates sprite anchored to center of bottom edge.
     * @param x x coordinate of sprite.
     * @param y y coordinate of sprite.
     * @param z z coordinate of sprite.
     * @param height height of sprite.
     * @param width width of sprite.
     * @param file string specifying the path of sprite.
     */
    public StaticSprite(double x, double y, double z, double height, double width, String file)
    {
        img = new ImageView(file);

        setX(x);
        setY(y);
        setZ(z);

        setHeight(height);
        setWidth(width);
    }

    /**
     * Method makes sprite rotate with scenes camera, simulating old school sprites.
     * @param follow whether sprite rotates with camera.
     */
    public void rotateWithCamera(boolean follow)
    {
        if (follow)
        {
            Rotate rotateY = new Rotate();
            rotateY.setAxis(Rotate.Y_AXIS);
            rotateY.setPivotY(getHeight() / 2d);
            rotateY.setPivotX(getWidth() / 2d);
            rotateY.setAngle(0);


            Rotate rotateX = new Rotate();
            rotateX.setAxis(Rotate.X_AXIS);
            rotateX.setPivotY(getHeight() / 2d);
            rotateX.setPivotX(getWidth() / 2d);
            rotateX.setAngle(0);

            if (Renderer.getCamera() != null)
                for (Transform t: Renderer.getCamera().getTransforms())
                {
                    Rotate rotate = (Rotate) t;

                    if (rotate.getAxis().equals(Rotate.Y_AXIS))
                        rotateY.setAngle(rotate.getAngle());

                    else if (rotate.getAxis().equals(Rotate.X_AXIS))
                        rotateX.setAngle(rotate.getAngle());
                }

            Renderer.addRotating(img);
            img.getTransforms().add(rotateY);
            img.getTransforms().add(rotateX);
        }
        else
        {
            Renderer.removeRotating(img);
            img.getTransforms().removeIf(t -> t instanceof Rotate);
        }
    }

    /**
     * Hides sprite in list of nodes specified.
     * @param siblings list of children of parent node.
     */
    public void hide(List<Node> siblings) { siblings.remove(img); }

    /**
     * Shows sprite in list of nodes specified.
     * @param siblings list of children of parent node.
     */
    public void show(List<Node> siblings) { siblings.add(img); }
}
