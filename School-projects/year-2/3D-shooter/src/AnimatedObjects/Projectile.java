package AnimatedObjects;
import javafx.scene.Node;
import javafx.scene.shape.Box;
import javafx.scene.shape.Sphere;

import java.io.File;
import java.util.List;

import Enemies.Enemy;
import Game.Game;
import Game.Player;
import Game.Renderer;
import Sprite.AnimatedSprite;

/**
 * Defines a floating orb which can hurt enemies or player, which gets destroyed on impact
 */
public class Projectile implements Updatable
{
    private enum Status {COLLISION, SPAWNING, DEAD, ALIVE}
    final private static List<Integer> SPAWN_ANIM_COLUMNS = List.of(1, 2, 3, 4, 5, 1, 2, 3, 4, 5);
    final private static List<Integer> SPAWN_ANIM_ROWS = List.of(1, 1, 1, 1, 1, 2, 2, 2, 2, 2);
    final private static List<Integer> IDLE_ANIM_COLUMNS = List.of(4, 4, 5, 5);
    final private static List<Integer> IDLE_ANIM_ROWS = List.of(2, 2, 2, 2);
    final private static List<Integer> HIT_ANIM_COLUMNS = List.of(5, 4, 3, 2, 1, 5, 4, 3, 2, 1);
    final private static List<Integer> HIT_ANIM_ROWS = List.of(2, 2, 2, 2, 2, 1, 1, 1, 1, 1);

    final static double ORB_WIDTH = 20;
    final static double ORB_DAMAGE = 20;
    final static double ORB_HORROR = 10;

    private double dx, dy, dz;
    private double x, y ,z;
    private Status status;
    private boolean inactive = false;
    private final boolean enemyProjectile;
    final private Sphere HIT_BOX = new Sphere(ORB_WIDTH / 2);

    private AnimatedSprite orb;

    /**
     * @return whether orb has collided and finished it's animations
     */
    @Override
    public boolean isInactive() { return inactive; }

    /**
     * Creates and displays floating orb which damages enemies or player.
     * @param x defines x coordinate of the middle of spawned orb.
     * @param y defines y coordinate of the middle of spawned orb.
     * @param z defines z coordinate of the middle of spawned orb.
     * @param dx defines displacement of orb in relation to x coordinate after each move.
     * @param dy defines displacement of orb in relation to y coordinate after each move.
     * @param dz defines displacement of orb in relation to z coordinate after each move.
     * @param enemyProjectile defines whether the orb is cast by enemy or player.
     */
    public Projectile(double x, double y, double z, double dx, double dy, double dz, boolean enemyProjectile)
    {
        this.dx = dx;
        this.dy = dy;
        this.dz = dz;
        this.x = x;
        this.y = y;
        this.z = z;
        this.enemyProjectile = enemyProjectile;
        try
        {
            orb = new AnimatedSprite(x, y + (ORB_WIDTH / 2d), z, ORB_WIDTH, ORB_WIDTH, 5, 2, "Assets" + File.separator + "sprites" + File.separator + "orb_anims.png");
            orb.rotateWithCamera(true);
            orb.show(Renderer.getGroup().getChildren());
            orb.setDuration(50);
            spawn();
        }
        catch (Exception e) { e.printStackTrace(); }

        HIT_BOX.setTranslateX(x);
        HIT_BOX.setTranslateZ(z);
        HIT_BOX.setTranslateY(y);
        //Game.Renderer.getGroup().getChildren().add(HIT_BOX);
    }

    private void move()
    {
        x += dx;
        y += dy;
        z += dz;

        orb.setY(y + (ORB_WIDTH / 2d));
        orb.setX(x);
        orb.setZ(z);

        if (projectileCollision())
        {
            x -= dx;
            y -= dy;
            z -= dz;

            dx = 0;
            dy = 0;
            dz  = 0;

            orb.setY(y + (ORB_WIDTH / 2d));
            orb.setX(x);
            orb.setZ(z);

            status = Status.COLLISION;
        }
        HIT_BOX.setTranslateX(x);
        HIT_BOX.setTranslateZ(z);
        HIT_BOX.setTranslateY(y);
    }

    private void setMove()
    {
        status = Status.ALIVE;
        orb.setAnimFrames(IDLE_ANIM_COLUMNS, IDLE_ANIM_ROWS);
        orb.animate(true);
    }

    private void spawn()
    {
        status = Status.SPAWNING;
        orb.setAnimFrames(SPAWN_ANIM_COLUMNS, SPAWN_ANIM_ROWS);
        orb.animate(false);
    }

    private void deSpawn()
    {
        status = Status.DEAD;
        orb.stopAnim();
        orb.setAnimFrames(HIT_ANIM_COLUMNS, HIT_ANIM_ROWS);
        orb.animate(false);
    }

    /**
     * Checks if spawned orb is too far from player or has hit enemy or player depending on whether orb was spawned by enemy or player
     * @return whether orb has collided
     */
    public boolean projectileCollision()
    {
        if (x < Player.getX() - 1000 || x > Player.getX() + 1000)
            return true;
        if (y < Player.getY() - 1000 || y > Player.getY() + 1000)
            return true;
        if (z < Player.getZ() - 1000 || z > Player.getZ() + 1000)
            return true;

        for (Node n: Renderer.getGroup().getChildren())
        {
            if (n instanceof Box)
            {
                Box b = (Box) n;
                if (b.getBoundsInParent().intersects(HIT_BOX.getBoundsInParent()))
                {
                    return true;
                }
            }
        }

        if (enemyProjectile)
        {
            if (HIT_BOX.getBoundsInParent().intersects(Player.getHitBox().getBoundsInParent()))
            {
                Player.dealDamage(ORB_DAMAGE);
                Player.dealHorror(ORB_HORROR);
                return true;
            }
        }
        else
        {
            Enemy enemy = Game.collidingEnemy(HIT_BOX.getBoundsInParent());
            if (enemy != null)
            {
                enemy.dealDamage(ORB_DAMAGE);
                return true;
            }
        }

        return false;
    }

    /**
     * Updates animations of orb depending whether it is spawning, moving or colliding.
     * Moves in direction described by dx, dy, dz parameters
     */
    @Override
    public void update()
    {
        if (inactive)
            return;

        switch (status)
        {
            case SPAWNING ->
                    {
                        if (orb.hasFinished())
                            setMove();
                    }
            case ALIVE -> move();
            case COLLISION -> deSpawn();
            case DEAD ->
                    {
                        if (orb.hasFinished())
                        {
                            inactive = true;
                            orb.hide(Renderer.getGroup().getChildren());
                        }

                    }
        }
    }


}
