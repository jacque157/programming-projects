package Weapons;

import Sprite.AnimatedSprite;
import javafx.scene.input.MouseEvent;

import java.io.File;
import java.util.List;

import Game.*;

/**
 * Represents a revolver player can shoot.
 */
public class Revolver implements Weapon
{
    private static int ammoLoaded = 6;
    private static AnimatedSprite revolverAnim;

    final private static int CLIP_SIZE = 6;
    final private static int MAX_AMMO = 18;
    final private static double DAMAGE = 10;
    final private static List<Integer> RELOAD_ANIM_COLUMNS = List.of(1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6);
    final private static List<Integer> RELOAD_ANIM_ROWS = List.of(2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5);
    final private static List<Integer> FIRE_ANIM_COLUMNS = List.of(2, 3, 4, 5, 6);
    final private static List<Integer> FIRE_ANIM_ROWS = List.of(1, 1, 1, 1, 1);
    final private static double IMAGE_HEIGHT = Renderer.SCENE_HEIGHT / 3;
    final private static double IMAGE_WIDTH = Renderer.SCENE_WIDTH / 8;

    /**
     * Initializes revolver sprite.
     */
    public Revolver()
    {
        try
        {
            revolverAnim = new AnimatedSprite(Renderer.SCENE_WIDTH / 2, Renderer.SCENE_HEIGHT, 0, IMAGE_HEIGHT, IMAGE_WIDTH,
                    6, 5, "Assets" + File.separator + "sprites" + File.separator  + "revolver_anims.png");
            revolverAnim.setDuration(100);
        }
        catch (Exception e) { e.printStackTrace(); }
    }

    /**
     * @return the damage dealt by bullet.
     */
    @Override
    public double getDamage() { return DAMAGE; }

    /**
     * @return the name of weapon.
     */
    @Override
    public String getName() { return " Revolver   ";}

    /**
     * @return munition loaded.
     */
    @Override
    public int getAmmoLoaded() { return ammoLoaded; }

    /**
     * @return the amount the chamber can hold.
     */
    @Override
    public int getClipSize() { return CLIP_SIZE; }

    /**
     * Hide revolver from GUI.
     */
    @Override
    public void hide() { revolverAnim.hide( GUI.getScreen().getChildren()); }

    /**
     * Show revolver in GUI.
     */
    @Override
    public void show() { revolverAnim.show( GUI.getScreen().getChildren()); }

    /**
     * @return maximum animation which can be carried.
     */
    @Override
    public int getMaxAmmo() { return MAX_AMMO; }

    /**
     * @return whether the animation stopped playing.
     */
    @Override
    public boolean isReady() { return revolverAnim.hasFinished(); }

    /**
     * Subtracts ammo in chamber and plays appropriate animation if able.
     * @param event mouse event.
     */
    @Override
    public void fire(MouseEvent event)
    {
        if (ammoLoaded == 0 || ! revolverAnim.hasFinished())
            return;

        ammoLoaded--;
        revolverAnim.setAnimFrames(FIRE_ANIM_COLUMNS, FIRE_ANIM_ROWS);
        revolverAnim.animate(false);
        GUI.updateWeaponGUI();
    }

    /**
     * Subtracts ammo carried and plays appropriate animation if able.
     */
    @Override
    public void reload()
    {
        if (Player.getRevolverAmmo() == 0 || ! revolverAnim.hasFinished())
            return;

        if (ammoLoaded == CLIP_SIZE)
            return;

        for (; 0 < Player.getRevolverAmmo(); Player.setRevolverAmmo(Player.getRevolverAmmo() - 1), ammoLoaded++)
            if (ammoLoaded == CLIP_SIZE)
                break;

        revolverAnim.setAnimFrames(RELOAD_ANIM_COLUMNS, RELOAD_ANIM_ROWS);
        revolverAnim.animate(false);
        GUI.updateWeaponGUI();
    }

    /**
     * @return ammo loaded and chamber size.
     */
    @Override
    public String munitionLoadedInfo() { return "" + ammoLoaded + " / " + CLIP_SIZE; }

    /**
     * @return ammo carried and max ammo which can be carried.
     */
    @Override
    public String munitionInfo() { return "" + Player.getRevolverAmmo() + " / " + MAX_AMMO; }

}
