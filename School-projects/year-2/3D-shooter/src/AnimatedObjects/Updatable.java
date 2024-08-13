package AnimatedObjects;

/**
 * Interface for classes which are supposed to be updated regularly in game cycle
 */
public interface Updatable
{
    /**
     * Method which describes whether an object needs not to be updated e.g. dead enemies are not active.
     * @return this object's status
     */
    boolean isInactive();

    /**
     * Update's parameters of object
     */
    void update();
}
