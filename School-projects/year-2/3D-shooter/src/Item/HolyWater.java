package Item;

import javafx.geometry.Bounds;
import javafx.scene.shape.Box;

import java.io.File;

import Game.*;
import Sprite.StaticSprite;

/**
 * Represents HolyWater item which is used to restore players sanity.
 */
public class HolyWater implements PickUp
{
    private static final double FLASK_WIDTH = 8;
    private static final double FLASK_HEIGHT = 8;
    private static final int HEALING_AMOUNT = 70;

    private StaticSprite medKit;
    private final Box HIT_BOX = new Box(Game.UNIT, Game.UNIT, Game.UNIT);

    private boolean active = true;

    /**
     * @return whether the item has been picked up.
     */
    @Override
    public boolean isActive() { return active; }

    /**
     * Places holy water item, represented by sprite, in scene.
     * @param x x coordinate of item in scene.
     * @param y y coordinate of item in scene.
     * @param z z coordinate of item in scene.
     */
    public HolyWater(double x, double y, double z)
    {
        medKit = new StaticSprite(x, y, z, FLASK_HEIGHT, FLASK_WIDTH, "Assets" + File.separator + "sprites" + File.separator + "holywater.png");
        medKit.rotateWithCamera(true);
        medKit.show(Renderer.getGroup().getChildren());

        HIT_BOX.setTranslateX(x);
        HIT_BOX.setTranslateZ(z);
        HIT_BOX.setTranslateY(y - (Game.UNIT / 2d) - 1);
    }

    /**
     * Heals players sanity and removes sprite from scene.
     * Updates weapon info if player is casting spells.
     */
    @Override
    public void pickUp()
    {
        if (isActive())
        {
            Player.heallHorror(HEALING_AMOUNT);
            medKit.hide(Renderer.getGroup().getChildren());
            GUI.updateWeaponGUI();
        }
        active = false;
    }

    /**
     * @param hitBoxBounds hitBox of entity.
     * @return whether the item is colliding with entity specified.
     */
    @Override
    public boolean collision(Bounds hitBoxBounds)
    {
        return HIT_BOX.getBoundsInParent().intersects(hitBoxBounds);
    }
}
