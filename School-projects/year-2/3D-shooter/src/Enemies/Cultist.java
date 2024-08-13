package Enemies;


import javafx.scene.shape.Box;
import java.io.File;
import java.util.List;
import java.util.concurrent.Semaphore;

import AnimatedObjects.Projectile;
import Game.*;
import Sprite.AnimatedSprite;
import Weapons.*;

/**
 * Class represents cultist type enemy handles it's movement and ai.
 */
public class Cultist extends Enemy
{
    private enum Status {IDLE, WALKING, DYING, CASTING}

    final private static List<Integer> CAST_ANIM_COLUMNS = List.of(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6);
    final private static List<Integer> CAST_ANIM_ROWS = List.of(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    final private static List<Integer> IDLE_ANIM_COLUMNS = List.of(1);
    final private static List<Integer> IDLE_ANIM_ROWS = List.of(3);
    final private static List<Integer> DEATH_ANIM_COLUMNS = List.of(1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6);
    final private static List<Integer> DEATH_ANIM_ROWS = List.of(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
    final private static List<Integer> WALK_ANIM_COLUMNS = List.of(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1);
    final private static List<Integer> WALK_ANIM_ROWS = List.of   (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4);

    final private static double CULTIST_WIDTH = 30;
    final private static double CULTIST_HEIGHT = 43;
    final private static double MAX_HP = 80;
    final private static double SPEED = 1;
    final private static int MAX_DISTANCE = 10;
    final private static Semaphore mutex = new Semaphore(1);

    private boolean exhausted = false;

    final private Box HIT_BOX = new Box(CULTIST_WIDTH, CULTIST_HEIGHT, CULTIST_WIDTH);

    private double hp;
    private Cultist.Status status;
    private Direction direction = Direction.IDLE;
    private boolean inactive = false;
    private int ticks = 0;
    final private int COOL_DOWN = 80;
    private AnimatedSprite cultist;

    /**
     * @return Whether cultist has died.
     */
    @Override
    public boolean isInactive() { return inactive; }

    /**
     * @return tiles cultist can visit.
     */
    @Override
    List<Character> getAllowedSpaces() { return List.of('.', 'D'); }

    /**
     * @return direction of movement.
     */
    @Override
    public Direction getDirection() { return direction; }

    /**
     * @return x coordinate of cultist in scene.
     */
    @Override
    public double getX() { return cultist.getX(); }

    /**
     * @return y coordinate of cultist in scene.
     */
    @Override
    public double getY() { return cultist.getY(); }

    /**
     * @return z coordinate of cultist in scene.
     */
    @Override
    public double getZ() { return cultist.getZ(); }

    /**
     * @return width of sprite representing enemy.
     */
    @Override
    public double getWidth() { return CULTIST_WIDTH; }

    /**
     * @return height of sprite representing enemy.
     */
    @Override
    public double getHeight() { return CULTIST_HEIGHT; }

    /**
     * @return the max level of breadth first search when searching for player on map.
     */
    @Override
    public int getMaxDistance() { return MAX_DISTANCE; }

    /**
     * @return the speed cultist moves each move.
     */
    @Override
    public double getSpeed() { return SPEED; }

    /**
     * @return box representing space occupied by cultist.
     */
    @Override
    public Box getHIT_BOX() { return HIT_BOX; }

    /**
     * Spawns cultist in Scene.
     * @param x x coordinate in scene.
     * @param y y coordinate in scene.
     * @param z z coordinate in scene.
     */
    public Cultist(double x, double y, double z)
    {
        hp = MAX_HP;
        status = Status.IDLE;
        try
        {
            cultist = new AnimatedSprite(x, y, z, CULTIST_HEIGHT, CULTIST_WIDTH, 6, 4, "Assets" + File.separator + "sprites" + File.separator + "cultist_anims.png");
            cultist.rotateWithCamera(true);
            cultist.show(Renderer.getGroup().getChildren());
            cultist.setDuration(50);
        }
        catch (Exception e) { e.printStackTrace(); }

        HIT_BOX.setTranslateX(x);
        HIT_BOX.setTranslateZ(z);
        HIT_BOX.setTranslateY(y - (CULTIST_HEIGHT / 2) - 1);
        cultist.getImage().setOnMouseClicked(event ->
        {
            Weapon weapon = Player.getActiveWeapon();
            if (weapon instanceof Revolver && weapon.getAmmoLoaded() > 0 && weapon.isReady())
            {
                dealDamage(Player.getActiveWeapon().getDamage());
            }
        });
    }

    /**
     * Deals damage to player.
     * @param damage amount of damage dealt by enemy.
     */
    @Override
    public void dealDamage(double damage)
    {
        try
        {
            mutex.acquire();
            hp -= damage;
            mutex.release();
        }
        catch (InterruptedException ignored) { }
    }

    private void adjust()
    {
        if (adjusted())
            return;

        int column = (int)Math.floor((getX())/ Game.UNIT);
        int row = (int)Math.floor((getZ()) / Game.UNIT);

        double x = column * Game.UNIT + (Game.UNIT / 2d);
        double z = row * Game.UNIT + (Game.UNIT / 2d);

        followPoint(x, z);
    }

    private void move(Direction direction)
    {
        this.direction = direction;
        double dx = dx(direction), dz = dz(direction);
        move(dx, dz);
    }

    private void move(double dx, double dz)
    {
        if (dx != 0)
        {
            cultist.setX(getX() + dx);
            HIT_BOX.setTranslateX(getX());
        }
        if (dz != 0)
        {
            cultist.setZ(getZ() + dz);
            HIT_BOX.setTranslateZ(getZ());
        }
    }

    private void attack()
    {
        if ( ! exhausted && cultist.getFrame() >= 12 && status == Status.CASTING)
        {
            exhausted = true;

            double x1 = getX(), z1 = getZ();
            double x2 = Player.getX(), z2 = Player.getZ();

            double k = (z2 - z1) / (x2 - x1);
            double alpha = Math.atan(k);

            if (x2 < x1 && z2 < z1)
                alpha += Math.PI;
            else if (x2 > x1 && z2 < z1)
                alpha += 2 * Math.PI;
            else if (x2 < x1 && z2 > z1)
                alpha += Math.PI;

            double x = getX() + Math.cos(alpha) * 8;
            double z = getZ() + Math.sin(alpha) * 8;
            double dx = 8 * Math.cos(alpha);
            double dz = 8 * Math.sin(alpha);
            Game.addObject(new Projectile(x, CULTIST_HEIGHT / -2, z, dx, 0, dz, true));
        }
    }

    private void playMove()
    {
        if (status != Status.WALKING)
        {
            status = Status.WALKING;
            cultist.stopAnim();
            cultist.setAnimFrames(WALK_ANIM_COLUMNS, WALK_ANIM_ROWS);
        }
        cultist.nextFrame(true);
    }

    private void playIdle()
    {
        if (status != Status.IDLE)
        {
            status = Status.IDLE;
            cultist.stopAnim();
            cultist.setAnimFrames(IDLE_ANIM_COLUMNS, IDLE_ANIM_ROWS);
        }
        cultist.nextFrame(true);
    }

    private void playDeath()
    {
        if (status != Status.DYING)
        {
            status = Status.DYING;
            cultist.stopAnim();
            cultist.setAnimFrames(DEATH_ANIM_COLUMNS, DEATH_ANIM_ROWS);
        }
        cultist.nextFrame(false);
    }

    private void playAttack()
    {
        if (status != Status.CASTING)
        {
            status = Status.CASTING;
            cultist.stopAnim();
            cultist.setAnimFrames(CAST_ANIM_COLUMNS, CAST_ANIM_ROWS);
        }
        cultist.nextFrame(false);
    }

    /**
     * Updates cultist's animations, position in scene and orb spawning.
     */
    @Override
    public void update()
    {
        if (inactive)
        {
            return;
        }

        if ( status == Status.DYING && ! cultist.hasFinished())
        {
            playDeath();
        }
        else if (status == Status.DYING)
        {
            inactive = true;
            return;
        }

        ticks++;

        if ( status == Status.CASTING && ! cultist.hasFinished() )
        {
            attack();
            playAttack();
            return;
        }
        else if (status == Status.CASTING )
        {
            playIdle();
            exhausted = false;
        }

        if (hp <= 0)
        {
            playDeath();
        }
        else if (lineOfSight())
        {
            if (ticks > COOL_DOWN)
            {
                direction = Direction.IDLE;
                playAttack();
                ticks = 0;
            }
        }
        else if (direction == Direction.IDLE && ! adjusted())
        {
            adjust();
            playMove();
        }
        else
        {
            Direction direction = findDirection();
            if (direction == Direction.IDLE) { playIdle(); }
            else { playMove(); }
            move(direction);
        }
    }

    private void followPoint(double x, double z)
    {
        double xs = getX(), zs = getZ();

        double k = (z - zs) / (x - xs);
        double alpha = Math.abs(Math.atan(k));

        double dx = Math.abs(Math.cos(alpha) * SPEED);
        double dz = Math.abs(Math.sin(alpha) * SPEED);

        if (xs > x && zs > z)
        {
            move(-dx, -dz);
        }
        else if (xs < x && zs > z)
        {
            move(dx, -dz);
        }
        else if (xs > x && zs < z)
        {
            move(-dx, dz);
        }
        else if (xs < x && zs < z)
        {
            move(dx, dz);
        }
    }
}

