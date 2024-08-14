package Game;

import javafx.geometry.Bounds;
import javafx.scene.Camera;
import javafx.scene.Node;
import javafx.scene.input.KeyCode;
import javafx.scene.shape.Box;
import javafx.scene.shape.Cylinder;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Semaphore;

import Item.PickUp;
import Weapons.*;

/**
 * Static class representing playable character
 */
public class Player
{
    private static double SPEED = 1;

    private enum Dir {FORWARD, BACKWARD, STOP, RIGHT, LEFT}

    private static final double FARCLIP = 500;
    private static final double HEIGHT = 30;
    private static final double WIDTH = 12;
    private static final double MAX_REASON = 120;
    private static final double MAX_VITALITY = 100;
    private static final int MAX_REVOLVER_AMMO = 18;
    private static final Cylinder HIT_BOX = new Cylinder(WIDTH / 2, HEIGHT);
    private static final Semaphore mutex = new Semaphore(1);

    static private Map<Dir, Boolean> directions;
    static private double angle = 0;
    static private double x = 0;
    static private double y;
    static private double z = 0;
    static private double health = MAX_VITALITY;
    static private double sanity = MAX_REASON;
    static private List<Weapon> weapons = List.of(new Spell(), new Revolver());

    static private int activeWeaponIndex = 0;
    static private int revolverAmmo = 6;

    /**
     * @return amount of bullets carried.
     */
    public static int getRevolverAmmo()
    {
        return revolverAmmo;
    }

    /**
     * @return hitBox representing space occupied by player.
     */
    public static Cylinder getHitBox() { return HIT_BOX; }

    /**
     * @return maximum sanity player can have.
     */
    public static double getMaxReason() { return MAX_REASON; }

    /**
     * @return maximum health player can have.
     */
    public static double getMaxVitality() { return MAX_VITALITY; }

    /**
     * @return maximum bullets for revolver player can own.
     */
    public static int getMaxRevolverAmmo() { return MAX_REVOLVER_AMMO; }

    /**
     * @return health of player.
     */
    public static double getHealth() { return health; }

    /**
     * @return sanity of player.
     */
    public static double getSanity() { return sanity; }

    /**
     * @return currently equipped weapon.
     */
    public static Weapon getActiveWeapon() { return weapons.get(activeWeaponIndex); }

    /**
     * Sets revolver bullets carried by player.
     * @param revolverAmmo
     */
    public static void setRevolverAmmo(int revolverAmmo)
    {
        Player.revolverAmmo = revolverAmmo;
    }

    /**
     * Sets player's health and displays the change in hud.
     * @param health new health.
     */
    public static void setHealth(double health)
    {
        Player.health = health;
        GUI.getLbHealth().setText("Vitality: " + getHealth() + " / " + MAX_VITALITY);
    }

    /**
     * Sets player's sanity and displays the change in hud.
     * @param sanity new sanity.
     */
    public static void setSanity(double sanity)
    {
        Player.sanity = sanity;
        GUI.getLbSanity().setText("Reason: " + getSanity() + " / " + MAX_REASON);
    }

    /**
     * Changes the speed character moves in scene.
     * @param SPEED new speed.
     */
    public static void setSPEED(double SPEED)
    {
        Player.SPEED = SPEED;
    }

    /**
     * Sets angle around y axis of player.
     * @param angle
     */
    public static void setAngle(double angle) { Player.angle = angle; }

    /**
     * Sets player's x coordinate in scenes and moves camera correspondingly.
     * @param x
     */
    public static void setX(double x)
    {
        Camera camera = Renderer.getCamera();
        Player.x = x;
        camera.translateXProperty().set(x);
        getHitBox().setTranslateX(x);
    }

    /**
     * Sets player's y coordinate in scenes and moves camera correspondingly.
     * @param y
     */
    public static void setY(double y)
    {
        Camera camera = Renderer.getCamera();
        Player.y = y;
        camera.translateYProperty().set(y);
        getHitBox().setTranslateY(y - (HEIGHT / 2) - 1);
    }

    /**
     * Sets player's z coordinate in scenes and moves camera correspondingly.
     * @param z
     */
    public static void setZ(double z)
    {
        Camera camera = Renderer.getCamera();
        Player.z = z;
        camera.translateZProperty().set(z);
        getHitBox().setTranslateZ(z);
    }

    /**
     * Sets pressed key on keyboard.
     * @param key key pressed.
     */
    static void setDirection(KeyCode key)
    {
        switch (key)
        {
            case W -> directions.put(Dir.FORWARD, true);
            case S -> directions.put(Dir.BACKWARD, true);
            case A -> directions.put(Dir.LEFT, true);
            case D -> directions.put(Dir.RIGHT, true);
        }
    }

    /**
     * Unsets released key on keyboard.
     * @param key key released.
     */
    static void unsetDirection(KeyCode key)
    {
        switch (key)
        {
            case W -> directions.put(Dir.FORWARD, false);
            case S -> directions.put(Dir.BACKWARD, false);
            case A -> directions.put(Dir.LEFT, false);
            case D -> directions.put(Dir.RIGHT, false);
        }
    }

    /**
     * Subtracts the specified amount from player's health.
     * If the health is less or equal to 0 player loses.
     * @param damage damage dealt to player.
     */
    public static void dealDamage(double damage)
    {
        try
        {
            mutex.acquire();
            setHealth(getHealth() - damage);
            mutex.release();
        }
        catch (InterruptedException ignored) {}
        if (health <= 0)
            Game.lose(false);
    }

    /**
     * Switches the currently equipped weapon for the next in sequence or
     * the first one if it was the last. Hides the previous weapon and draws
     * the next, updates the player's HUD.
     */
    public static void switchWeapon()
    {
        getActiveWeapon().hide();
        activeWeaponIndex = (activeWeaponIndex + 1) % weapons.size();
        GUI.updateWeaponGUI();
        getActiveWeapon().show();
    }

    /**
     * Subtracts the specified amount from player's sanity.
     * If the sanity is less or equal to 0 player loses.
     * @param horror horror dealt to player.
     */
    public static void dealHorror(double horror)
    {
        try
        {
            mutex.acquire();
            setSanity(getSanity() - horror);
            mutex.release();
        }
        catch (InterruptedException ignored) {}

        if (sanity <= 0)
            Game.lose(true);
    }

    /**
     * Adds the specified amount to player's health.
     * Prevents player from having more health the maximum health specified.
     * @param amount amount healed.
     */
    public static void healDamage(double amount)
    {
        try
        {
            mutex.acquire();
            setHealth(Math.min(getHealth() + amount, MAX_VITALITY));
            mutex.release();
        }
        catch (InterruptedException ignored) {}
    }

    /**
     * Adds the specified amount to player's sanity.
     * Prevents player from having more sanity the maximum sanity specified.
     * @param amount amount restored.
     */
    public static void heallHorror(double amount)
    {
        try
        {
            mutex.acquire();
            setSanity(Math.min(getSanity() + amount, MAX_REASON));
            mutex.release();
        }
        catch (InterruptedException ignored) {}
    }

    /**
     * @return the angle of player's rotation along y axis.
     */
    public static double getAngle() { return angle; }

    /**
     * @return x coordinate in scene.
     */
    public static double getX() { return x; }

    /**
     * @return y coordinate in scene.
     */
    public static double getY() { return y; }

    /**
     * @return z coordinate in scene.
     */
    public static double getZ() { return z; }


    /**
     * Initializes player in scene.
     * @param x x coordinate in scene.
     * @param z z coordinate in scene.
     */
    static void init(double x, double z)
    {
        setX(x);
        setY(-HEIGHT);
        setZ(z);

        setSanity(MAX_REASON);
        setHealth(MAX_VITALITY);

        Renderer.getCamera().setFarClip(FARCLIP);

        directions = new HashMap<>();
        for(Dir direction : Dir.values())
            directions.put(direction, false);
        directions.put(Dir.STOP, true);

        switchWeapon();
        //Game.Renderer.getGroup().getChildren().add(getHitBox());
    }

    /**
     * Moves player in scene in relation to pressed/released keys.
     * Ends game if player has reached exit.
     */
    static void movement()
    {
        double dx = Math.sin(Math.toRadians(angle)) * SPEED;
        double dz = Math.cos(Math.toRadians(angle)) * SPEED;

        double oldZ = getZ();
        double oldX = getX();

        if (directions.get(Dir.FORWARD))
        {
            setZ(getZ() + dz);
            setX(getX() + dx);
        }

        if (directions.get(Dir.BACKWARD))
        {
            setZ(getZ() - dz);
            setX(getX() - dx);
        }

        if (directions.get(Dir.LEFT))
        {
            setZ(getZ() + dx);
            setX(getX() - dz);
        }

        if (directions.get(Dir.RIGHT))
        {
            setZ(getZ() - dx);
            setX(getX() + dz);
        }

        if (checkPlayerCollision())
        {
            setZ(oldZ);
            setX(oldX);
        }

        if (Game.exitReached(getX(), getZ()))
            Game.win();
    }

    /**
     * Checks whether player is in vicinity of item which can be picked up.
     */
    public static void use()
    {
        PickUp item = Game.collidingPickUp(HIT_BOX.getBoundsInParent());
        if (item != null)
            item.pickUp();
    }

    /**
     * @return whether player is colliding with wall.
     */
    public static boolean checkPlayerCollision()
    {
        for (Node n: Renderer.getGroup().getChildren())
        {
            if (n instanceof Box)
            {
                Box b = (Box) n;
                if (b.getBoundsInParent().intersects(getHitBox().getBoundsInParent()))
                    return true;
            }
        }
        return false;
    }

    /**
     * @param hitBox entities hitBox to be considered in collision.
     * @return whether player has collided with hitBox specified.
     */
    public static boolean checkPlayerCollision(Bounds hitBox) { return HIT_BOX.getBoundsInParent().intersects(hitBox); }
}
