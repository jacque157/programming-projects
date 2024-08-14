package Game;

import javafx.scene.*;
import javafx.scene.paint.Color;
import javafx.scene.transform.Rotate;
import javafx.scene.transform.Transform;
import javafx.stage.Screen;

import java.util.ArrayList;
import java.util.List;

/**
 * Static class for representing 3d scene of game.
 */
public class Renderer
{
    /**
     * Height of scene displayed.
     */
    static public final double SCENE_HEIGHT = Screen.getPrimary().getBounds().getHeight();
    /**
     * Width of scene displayed.
     */
    static public final double SCENE_WIDTH = Screen.getPrimary().getBounds().getWidth();
    static private Group group;
    static private Camera camera;
    static private Scene scene;
    static private SubScene subScene;

    static private final List<Node> rotating = new ArrayList<>();

    /**
     * @return group of scene.
     */
    static public Group getGroup() { return group; }

    /**
     * @return camera of scene.
     */
    static public Camera getCamera() { return camera; }

    /**
     * @return scene of game.
     */
    static public Scene getScene() { return scene; }

    /**
     * @return subScene of game. The 3d scene itself.
     */
    public static SubScene getSubScene() { return subScene; }

    /**
     * Adds node to list of nodes which rotate with player's camera. Used to represent old school sprites.
     * @param element node to be added.
     */
    static public void addRotating(Node element)
    {
        rotating.add(element);
    }

    /**
     * Removes node from list of nodes which rotate with player's camera.
     * @param element node to be removed.
     */
    static public void removeRotating(Node element)
    {
        rotating.remove(element);
    }

    /**
     * Initializes scene and subScene of game.
     * @return scene of game.
     */
    static public Scene init()
    {
        Renderer.group = new Group();
        Renderer.camera = new PerspectiveCamera(true);

        Renderer.subScene = new SubScene(group, SCENE_WIDTH, SCENE_HEIGHT, true, SceneAntialiasing.BALANCED);
        Renderer.scene = new Scene(GUI.getLayout(), SCENE_WIDTH, SCENE_HEIGHT);

        subScene.setFill(Color.DARKSLATEGRAY);
        subScene.setCamera(camera);

        return scene;
    }

    /**
     * Rotates camera.
     * @param dx amount of rotation along y axis.
     * @param dy amount of rotation along x axis.
     */
    static public void rotateCamera(double dx, double dy)
    {
        ArrayList<Transform> transforms = new ArrayList<>(camera.getTransforms());
        camera.getTransforms().clear();

        for (Transform t: transforms)
        {
            Rotate rotate = (Rotate) t;

            if (rotate.getAxis().equals(Rotate.Y_AXIS))
            {
                rotate.setAngle(rotate.getAngle() + (EventsHandler.MOUSE_SENSITIVITY * (dx)));
                Player.setAngle(rotate.getAngle());
            }
            else if (rotate.getAxis().equals(Rotate.X_AXIS))
            {
                double newAngle = rotate.getAngle() - (EventsHandler.MOUSE_SENSITIVITY * (dy));
                if (newAngle > -75 && newAngle < 75)
                    rotate.setAngle(newAngle);
            }
            camera.getTransforms().add(rotate);
        }
        rotateNodes(dx, dy);
    }

    /**
     * Rotates nodes. Used to simulate old school sprites, which rotated with player's camera.
     * @param dx amount of rotation along y axis.
     * @param dy amount of rotation along x axis.
     */
    static private void rotateNodes(double dx, double dy)
    {
        for (Node element : rotating)
        {
            ArrayList<Transform> transforms = new ArrayList<>(element.getTransforms());
            element.getTransforms().clear();

            for (Transform t: transforms)
            {
                Rotate rotate = (Rotate) t;

                if (rotate.getAxis().equals(Rotate.Y_AXIS))
                {
                    rotate.setAngle(rotate.getAngle() + (EventsHandler.MOUSE_SENSITIVITY * (dx)));
                }
                else if (rotate.getAxis().equals(Rotate.X_AXIS))
                {
                    double newAngle = rotate.getAngle() - (EventsHandler.MOUSE_SENSITIVITY * (dy));
                    if (newAngle > -75 && newAngle < 75)
                        rotate.setAngle(newAngle);
                }
                element.getTransforms().add(rotate);
            }
        }
    }
}
