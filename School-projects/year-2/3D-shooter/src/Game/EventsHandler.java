package Game;

import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.scene.Camera;
import javafx.scene.Scene;
import javafx.scene.SubScene;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.input.MouseEvent;
import javafx.scene.robot.Robot;
import javafx.scene.transform.Rotate;
import javafx.stage.Screen;
import javafx.stage.Stage;
import javafx.util.Duration;
import java.util.concurrent.Semaphore;

/**
 * Static class handling every user's input.
 */
public class EventsHandler
{
    /**
     * Mouse sensitivity represents how much camera rotates on mouse movement.
     */
    static public final float MOUSE_SENSITIVITY = 0.01f;

    static private final Robot robot = new Robot();
    static private final Semaphore mutex = new Semaphore(1);
    static private Stage stage;

    /**
     * Initializes every input expected from player, sets player's camera, handles player's movement.
     * @param stage
     */
    static void init(Stage stage)
    {
        EventsHandler.stage = stage;
        Scene scene = Renderer.getScene();
        SubScene subScene = Renderer.getSubScene();

        Camera camera = Renderer.getCamera();

        scene.setOnKeyPressed(EventsHandler::buttonPressed);

        scene.setOnKeyReleased(EventsHandler::buttonReleased);

        subScene.setPickOnBounds(true);
        scene.setOnMouseClicked(EventsHandler::mouseClicked);

        camera.getTransforms().add(new Rotate(0, 0,0,0, Rotate.Y_AXIS));
        camera.getTransforms().add(new Rotate(0, 0,0,0, Rotate.X_AXIS));

        scene.setOnMouseMoved(EventsHandler::mouseMoved);

        Timeline timer = new Timeline(new KeyFrame(new Duration(30), event ->
        {
            centerMouse();
            Player.movement();
        }));
        timer.setCycleCount(Timeline.INDEFINITE);
        timer.play();
    }

    /**
     * Method centers mouse for purposes of camera rotation.
     * Prevents mouse listener from picking up mouse movement.
     */
    static void centerMouse()
    {
        if ( ! Game.isPaused() && EventsHandler.stage.isFocused() && mutex.tryAcquire())
        {
            robot.mouseMove(
                    (int)(Screen.getPrimary().getBounds().getWidth() / 2),
                    (int)(Screen.getPrimary().getBounds().getHeight() / 2));
            mutex.release();
        }
    }

    /**
     * Calls appropriate method for player's keyboard input.
     * R -> reload gun.
     * SHIFT -> sprint.
     * E -> pickUp / use.
     * ESC -> close application.
     * A, W, S, D -> movement.
     * @param event keyboard event.
     */
    static void buttonPressed(KeyEvent event)
    {
        switch (event.getCode())
        {
            case R ->
                    {
                        if (Player.getActiveWeapon().isReady())
                            Player.getActiveWeapon().reload();
                    }
            case SHIFT -> Player.setSPEED(2);
            case E -> Player.use();
            case ESCAPE -> stage.close();
            default -> Player.setDirection(event.getCode());
        }
    }

    /**
     * Calls appropriate method for player's keyboard input.
     * SHIFT -> stop sprint.
     * A, W, S, D -> stop movement.
     * @param event keyboard event.
     */
    static void buttonReleased(KeyEvent event)
    {
        //Game.printMap();
        if (event.getCode() == KeyCode.SHIFT)
            Player.setSPEED(1);
        else
            Player.unsetDirection(event.getCode());
    }

    /**
     * Rotates player's camera depending on distance the mouse travelled from center of screen.
     * @param event mouse event.
     */
    static void mouseMoved(MouseEvent event)
    {
        if (mutex.tryAcquire())
        {
            double x = robot.getMouseX(), y = robot.getMouseY();
            double x0 = Screen.getPrimary().getBounds().getWidth() / 2,
                    y0 = Screen.getPrimary().getBounds().getHeight() / 2;

            Renderer.rotateCamera(x - x0, y - y0);

            mutex.release();
        }
    }

    /**
     * Calls appropriate method for player's mouse input.
     * Left mouse button -> shot weapon / cast spell.
     * right mouse button -> switch weapon
     * @param event mouse event
     */
    static void mouseClicked(MouseEvent event)
    {
        switch (event.getButton())
        {
            case PRIMARY ->
                    {
                        if (Player.getActiveWeapon().isReady())
                            Player.getActiveWeapon().fire(event);
                    }
            case SECONDARY ->
                    {
                        if (Player.getActiveWeapon().isReady())
                            Player.switchWeapon();
                    }
        }
    }
}
