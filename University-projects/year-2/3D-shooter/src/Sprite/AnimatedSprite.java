package Sprite;

import javafx.animation.Animation;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.geometry.Rectangle2D;
import javafx.util.Duration;

import java.util.List;

/**
 * Class extends sprite, provides methods for animating.
 */
public class AnimatedSprite extends StaticSprite
{
    final private double frameWidth;
    final private double frameHeight;

    private int frame = 0;
    private Timeline timer;
    private double duration = 30;
    private boolean automatic = false;

    private List<Integer> animColumns, animRows;

    /**
     * @return currently played frame.
     */
    public int getFrame() { return frame; }

    /**
     * Sets sequence of played frames.
     * @param columns columns of frames to be played in order.
     * @param rows rows of frames to be played in order.
     */
    public void setAnimFrames(List<Integer> columns, List<Integer> rows)
    {
        animColumns = columns;
        animRows= rows;
        frame = 0;
    }

    /**
     * Sets period in milliseconds of animation.
     * @param duration milliseconds.
     */
    public void setDuration(double duration) { this.duration = duration; }

    /**
     * Creates animated sprite from large image consisting of evenly spaced frames.
     * @param x x coordinate of sprite.
     * @param y y coordinate of sprite.
     * @param z z coordinate of sprite.
     * @param height height of sprite.
     * @param width width of sprite.
     * @param columns how many sprites are in each row.
     * @param rows how many sprites are in each column.
     * @param file string representing path of image.
     * @throws Exception if height or columns are below 1.
     */
    public AnimatedSprite(double x, double y, double z, double height, double width, int columns, int rows, String file) throws Exception
    {
        super(x, y, z, file);
        if (columns < 1 || rows < 1)
            throw new Exception("Number of columns and rows must be greater than 0");

        frameWidth = getWidth() / columns;  // viewport requires width and height of original image
        frameHeight = getHeight() / rows;

        var rec =  new Rectangle2D(0, 0, frameWidth, frameHeight);
        getImage().setViewport(rec);

        setHeight(height);
        setWidth(width);
        setX(x);
        setY(y);
    }

    /**
     * Creates animated sprite from large image consisting of evenly spaced frames.
     * @param x x coordinate of sprite.
     * @param y y coordinate of sprite.
     * @param z z coordinate of sprite.
     * @param columns how many sprites are in each row.
     * @param rows how many sprites are in each column.
     * @param file string representing path of image.
     * @throws Exception if height or columns are below 1.
     */
    public AnimatedSprite(double x, double y, double z, int columns, int rows, String file) throws Exception
    {
        super(x, y, z, file);
        if (columns < 1 || rows < 1)
            throw new Exception("Number of columns and rows must be greater than 0");

        frameWidth = getWidth() / columns;
        frameHeight = getHeight() / rows;

        setWidth(getWidth() / columns);
        setHeight(getHeight() / rows);
        getImage().setViewport(new Rectangle2D(0, 0, frameWidth, frameHeight));
        setX(x);
        setY(y);
    }

    /**
     * Changes sprite to next in animation
     * @param loop whether the animation starts over after last frame.
     * @return whether the frame was switched successfully.
     */
    public boolean nextFrame(boolean loop)
    {
        if (automatic)
            return false;

        if (animColumns == null || animRows == null)
            return false;

        if (animColumns.size() == 0 || animRows.size() == 0)
            return false;

        if ( ! loop && (frame == animColumns.size() || frame == animRows.size()))
            return false;

        if (frame >= animColumns.size())
            frame = 0;

        if (frame >= animRows.size())
            frame = 0;

        int column = animColumns.get(frame);
        int row = animRows.get(frame);

        getImage().setViewport(new Rectangle2D((column - 1) * frameWidth, (row - 1) * frameHeight, frameWidth, frameHeight));
        frame++;

        return true;
    }

    /**
     * Starts animation in new TimeLine, which updates sprite every millisecond specified.
     * @param loop whether the animation starts over after last frame.
     */
    public void animate(boolean loop)
    {
        if (animColumns == null || animRows == null)
            return;

        if (animColumns.size() == 0 || animRows.size() == 0)
            return;

        if ( automatic && ! hasFinished())
            return;

        frame = 0;
        automatic = true;
        timer = new Timeline(new KeyFrame(new Duration(duration), event ->
        {
            if (frame >= animColumns.size())
                frame = 0;

            if (frame >= animRows.size())
                frame = 0;

            int column = animColumns.get(frame);
            int row = animRows.get(frame);

            getImage().setViewport(new Rectangle2D((column - 1) * frameWidth, (row - 1) * frameHeight, frameWidth, frameHeight));
            frame++;
        }));

        if (loop)
            timer.setCycleCount(Timeline.INDEFINITE);
        else
            timer.setCycleCount(animColumns.size());

        timer.play();
    }

    /**
     * @return whether last frame was played.
     */
    public boolean hasFinished()
    {
        if (animColumns == null || animRows == null)
            return true;
        if ( ! automatic )
            return frame == animColumns.size() || frame == animRows.size();
        return timer == null || timer.getStatus() == Animation.Status.STOPPED;
    }

    /**
     * Stops currently playing animation.
     */
    public void stopAnim()
    {
        if (timer != null)
        {
            timer.stop();
            automatic = true;
        }
    }

}
