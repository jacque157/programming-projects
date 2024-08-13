package Weapons;

import Sprite.AnimatedSprite;
import javafx.animation.Animation;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.scene.input.MouseEvent;
import javafx.scene.transform.Rotate;
import javafx.scene.transform.Transform;
import javafx.util.Duration;

import java.io.File;
import java.util.List;

import Game.*;
import AnimatedObjects.Projectile;

/**
 * Represents a spell player can cast.
 * Casting costs reason.
 */
public class Spell implements Weapon
{
    private static AnimatedSprite spellAnim;
    private static Timeline timer;

    final private static List<Integer> CAST_ANIM_COLUMNS = List.of(1, 2, 3, 4, 3, 2, 1);
    final private static List<Integer> CAST_ANIM_ROWS = List.of(1, 1, 1, 1, 1, 1, 1);
    final private static double IMAGE_HEIGHT = Renderer.SCENE_HEIGHT / 4;
    final private static double IMAGE_WIDTH = Renderer.SCENE_WIDTH / 3;
    final static double CAST_HORROR = 20;

    /**
     * Initializes casting sprite.
     */
    public Spell()
    {
        try
        {
            spellAnim = new AnimatedSprite(Renderer.SCENE_WIDTH - IMAGE_WIDTH / 2, Renderer.SCENE_HEIGHT, 0, IMAGE_HEIGHT, IMAGE_WIDTH,
                    4, 1, "Assets" + File.separator + "sprites" + File.separator + "cast_anim.png");
            spellAnim.setDuration(100);
        }
        catch (Exception e) { e.printStackTrace(); }
    }

    /**
     * @return damage dealt by spell.
     */
    @Override
    public double getDamage() { return 0; }

    /**
     * @return the name of spell.
     */
    @Override
    public String getName() { return "Blue Flame "; }

    /**
     * @return max ammo which can be carried.
     */
    @Override
    public int getMaxAmmo() { return 0; }

    /**
     * @return ammo loaded.
     */
    @Override
    public int getAmmoLoaded() { return 0; }

    /**
     * @return size of magazine.
     */
    @Override
    public int getClipSize() { return 0; }

    /**
     * @return Whether the player finished casting animation.
     */
    @Override
    public boolean isReady() { return timer == null || timer.getStatus() == Animation.Status.STOPPED; }

    /**
     * Hide casting hand in GUI.
     */
    @Override
    public void hide() { spellAnim.hide( GUI.getScreen().getChildren()); }

    /**
     * Show casting hand in GUI.
     */
    @Override
    public void show() { spellAnim.show( GUI.getScreen().getChildren()); }

    /**
     * Starts playing animation and deploys an orb in direction of mouse click which damages enemies.
     * Prevents player from killing them by sanity loss.
     * @param event mouse event.
     */
    @Override
    public void fire(MouseEvent event)
    {
        if ( ! isReady())
            return;

        if (Player.getSanity() - CAST_HORROR <= 0)
            return;

        Player.dealHorror(CAST_HORROR);
        spellAnim.setAnimFrames(CAST_ANIM_COLUMNS, CAST_ANIM_ROWS);
        spellAnim.animate(false);

        timer = new Timeline(new KeyFrame(new Duration(50), frame ->
        {
            if(spellAnim.getFrame() >= 4)
            {
                double alpha = 0;
                for (Transform t: Renderer.getCamera().getTransforms())
                    if (t instanceof Rotate)
                    {
                        Rotate rotate = (Rotate) t;
                        if (rotate.getAxis().equals(Rotate.X_AXIS))
                        {
                            alpha = rotate.getAngle();
                            break;
                        }
                    }

                double dy = Math.sin(Math.toRadians(alpha)) * 4;
                double x = Player.getX() + Math.sin(Math.toRadians(Player.getAngle())) * 8;
                double z = Player.getZ() + Math.cos(Math.toRadians(Player.getAngle())) * 8;
                double dx = 8 * Math.cos(Math.toRadians(alpha)) * Math.sin(Math.toRadians(Player.getAngle()));
                double dz = 8 * Math.cos(Math.toRadians(alpha)) * Math.cos(Math.toRadians(Player.getAngle()));

                Game.addObject(new Projectile(x, Player.getY(), z, dx, -dy, dz, false));
                timer.stop();
            }
        }));

        GUI.updateWeaponGUI();
        timer.setCycleCount(Animation.INDEFINITE);
        timer.play();
    }

    /**
     * Reloads weapon.
     */
    @Override
    public void reload() { }

    /**
     * @return string representing information about loaded munition.
     */
    @Override
    public String munitionLoadedInfo()
    {
        return "-/-";
    }

    /**
     * @return the amount of players sanity and maximum sanity.
     */
    @Override
    public String munitionInfo()
    {
        return Player.getSanity() + "/" + Player.getMaxReason();
    }
}
