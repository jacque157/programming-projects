package Item;

import javafx.geometry.Bounds;
import javafx.scene.shape.Box;

import java.io.File;

import Sprite.StaticSprite;
import Game.*;

/**
 * Represents RevolverAmmo item which is used to reload revolver.
 */
public class RevolverAmmo implements PickUp
{
    private static final double AMMO_WIDTH = 10;
    private static final double AMMO_HEIGHT = 5;
    private static final int AMMO_AMOUNT = 6;

    StaticSprite ammo;
    final Box HIT_BOX = new Box(Game.UNIT, Game.UNIT, Game.UNIT);

    private boolean active = true;

    /**
     * @return whether the item has been picked up.
     */
    @Override
    public boolean isActive() { return active; }

    /**
     * Places revolver bullets item, represented by sprite, in scene.
     * @param x x coordinate of item in scene.
     * @param y y coordinate of item in scene.
     * @param z z coordinate of item in scene.
     */
    public RevolverAmmo(double x, double y, double z)
    {
        ammo = new StaticSprite(x, y, z, AMMO_HEIGHT, AMMO_WIDTH, "Assets" + File.separator + "sprites" + File.separator + "handgun_ammo.png");
        ammo.rotateWithCamera(true);
        ammo.show(Renderer.getGroup().getChildren());

        HIT_BOX.setTranslateX(x);
        HIT_BOX.setTranslateZ(z);
        HIT_BOX.setTranslateY(y - (Game.UNIT / 2d) - 1);
    }

    /**
     * Adds bullets to players inventory and removes sprite from scene.
     * Method prevents player from having more ammo than they can carry.
     */
    @Override
    public void pickUp()
    {
        if (isActive())
        {
            Player.setRevolverAmmo(Math.min(Player.getRevolverAmmo() + AMMO_AMOUNT, Player.getMaxRevolverAmmo()));
            GUI.updateWeaponGUI();
            ammo.hide(Renderer.getGroup().getChildren());
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
