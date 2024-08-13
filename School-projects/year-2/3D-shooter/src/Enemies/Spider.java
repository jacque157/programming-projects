package Enemies;

import javafx.scene.shape.Box;
import java.io.File;
import java.util.*;
import java.util.concurrent.Semaphore;

import Game.*;
import Sprite.AnimatedSprite;
import Weapons.*;

/**
 * Class represents spider type enemy handles it's movement and ai.
 */
public class Spider extends Enemy
{
    private enum Status {IDLE, WALKING, DYING, ATTACKING}

    final private static List<Integer> BITE_ANIM_COLUMNS = List.of(1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6);
    final private static List<Integer> BITE_ANIM_ROWS = List.of(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    final private static List<Integer> IDLE_ANIM_COLUMNS = List.of(6);
    final private static List<Integer> IDLE_ANIM_ROWS = List.of(2);
    final private static List<Integer> DEATH_ANIM_COLUMNS = List.of(1, 1, 2, 2, 3, 3, 4, 4, 5, 5);
    final private static List<Integer> DEATH_ANIM_ROWS = List.of(2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
    final private static List<Integer> WALK_ANIM_COLUMNS = List.of(1, 2, 3, 4, 5, 6); // 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6
    final private static List<Integer> WALK_ANIM_ROWS = List.of(3, 3, 3, 3, 3, 3);  // 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3

    final private static double SPIDER_WIDTH = 40;
    final private static double SPIDER_HEIGHT = 30;
    final private static double MAX_HP = 35;
    final private static double DAMAGE = 20;
    final private static double HORROR = 15;
    final private static double SPEED = 2;
    final private static int MAX_DISTANCE = 8;
    final private static Semaphore mutex = new Semaphore(1);


    final private Box HIT_BOX = new Box(SPIDER_WIDTH, SPIDER_HEIGHT, SPIDER_WIDTH);

    private double hp;
    private Spider.Status status;
    private Direction direction = Direction.IDLE;
    private boolean inactive = false;

    private AnimatedSprite spider;

    /**
     * @return tiles spider can visit.
     */
    @Override
    public List<Character> getAllowedSpaces() { return List.of('.', 's', 'D'); }

    /**
     * @return direction of movement.
     */
    @Override
    public Direction getDirection() { return direction; }

    /**
     * @return x coordinate of spider in scene.
     */
    @Override
    public double getX() { return spider.getX(); }

    /**
     * @return y coordinate of spider in scene.
     */
    @Override
    public double getY() { return spider.getY(); }

    /**
     * @return z coordinate of spider in scene.
     */
    @Override
    public double getZ() { return spider.getZ(); }

    /**
     * @return width of sprite representing enemy.
     */
    @Override
    public double getWidth() { return SPIDER_WIDTH; }

    /**
     * @return height of sprite representing enemy.
     */
    @Override
    public double getHeight() { return SPIDER_HEIGHT; }

    /**
     * @return the max level of breadth first search when searching for player on map.
     */
    @Override
    public int getMaxDistance() { return MAX_DISTANCE; }

    /**
     * @return the speed spider moves each move.
     */
    @Override
    public double getSpeed() { return SPEED; }

    /**
     * @return box representing space occupied by spider.
     */
    @Override
    public Box getHIT_BOX() { return HIT_BOX; }

    /**
     * @return Whether spider has died.
     */
    @Override
    public boolean isInactive() { return inactive; }


    /**
     * Spawns spider in Scene.
     * @param x x coordinate in scene.
     * @param y y coordinate in scene.
     * @param z z coordinate in scene.
     */
    public Spider(double x, double y, double z)
    {
        hp = MAX_HP;
        status = Status.IDLE;
        try
        {
            spider = new AnimatedSprite(x, y, z, SPIDER_HEIGHT, SPIDER_WIDTH, 6, 3, "Assets" + File.separator + "sprites" + File.separator + "spider_anims.png");
            spider.rotateWithCamera(true);
            spider.show(Renderer.getGroup().getChildren());
        }
        catch (Exception e) { e.printStackTrace(); }

        HIT_BOX.setTranslateX(x);
        HIT_BOX.setTranslateZ(z);
        HIT_BOX.setTranslateY(y - (SPIDER_HEIGHT / 2) - 1);
        spider.getImage().setOnMouseClicked(event ->
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

        int column = (int)Math.floor((spider.getX())/ Game.UNIT);
        int row = (int)Math.floor((spider.getZ()) / Game.UNIT);

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
            spider.setX(spider.getX() + dx);
            HIT_BOX.setTranslateX(spider.getX());
        }
        if (dz != 0)
        {
            spider.setZ(spider.getZ() + dz);
            HIT_BOX.setTranslateZ(spider.getZ());
        }
    }

    private void attack()
    {
        if (spider.getFrame() == 12 && Player.checkPlayerCollision(HIT_BOX.getBoundsInParent()) && status == Status.ATTACKING)
        {
            Player.dealHorror(HORROR);
            Player.dealDamage(DAMAGE);
        }
    }

    private void playMove()
    {
        if (status != Status.WALKING)
        {
            status = Status.WALKING;
            spider.stopAnim();
            spider.setAnimFrames(WALK_ANIM_COLUMNS, WALK_ANIM_ROWS);
        }
        spider.nextFrame(true);
    }

    private void playIdle()
    {
        if (status != Status.IDLE)
        {
            status = Status.IDLE;
            spider.stopAnim();
            spider.setAnimFrames(IDLE_ANIM_COLUMNS, IDLE_ANIM_ROWS);
        }

        spider.nextFrame(true);
    }

    private void playDeath()
    {
        if (status != Status.DYING)
        {
            status = Status.DYING;
            spider.stopAnim();
            spider.setAnimFrames(DEATH_ANIM_COLUMNS, DEATH_ANIM_ROWS);
        }

        spider.nextFrame(false);
    }

    private void playAttack()
    {
        if (status != Status.ATTACKING)
        {
            status = Status.ATTACKING;
            spider.stopAnim();
            spider.setAnimFrames(BITE_ANIM_COLUMNS, BITE_ANIM_ROWS);
        }
        spider.nextFrame(false);
    }


    /**
     * Updates spider's animations and position in scene.
     */
    @Override
    public void update()
    {
        if (inactive)
        {
            return;
        }

        if ( status == Status.DYING && ! spider.hasFinished())
        {
            playDeath();
        }
        else if (status == Status.DYING)
        {
            inactive = true;
            return;
        }

        if ( status == Status.ATTACKING && ! spider.hasFinished() )
        {
            attack();
            playAttack();
            return;
        }
        else if (status == Status.ATTACKING)
        {
            playIdle();
        }

        if (hp <= 0)
        {
            playDeath();
        }
        else if (Player.checkPlayerCollision(getHIT_BOX().getBoundsInParent()))
        {
            playAttack();
        }
        else if (lineOfSight())
        {
            direction = Direction.IDLE;
            playMove();
            followPoint(Player.getX(), Player.getZ());
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

    void followPoint(double x, double z)
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
