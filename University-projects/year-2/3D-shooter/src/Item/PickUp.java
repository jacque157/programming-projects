package Item;

import javafx.geometry.Bounds;

/**
 * Represents item which can be picked up.
 */
public interface PickUp
{
    /**
     * @return whether item has been picked up.
     */
    boolean isActive();

    /**
     * Pick up item.
     */
    void pickUp();

    /**
     * @param hitBoxBounds hitBox of entity.
     * @return whether entity is colliding with item.
     */
    boolean collision(Bounds hitBoxBounds);
}
