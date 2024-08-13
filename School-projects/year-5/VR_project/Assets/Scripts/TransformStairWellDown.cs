using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TransformStairWellDown : MonoBehaviour
{
    public TorchPuzzle puzzleController;
    public GameObject EmptyStairWell;
    public GameObject StartHall;
    public GameObject EndHall;
    public GameObject EntryHall;

    public GameObject floorTransferObject;

    public GameObject unlockedDoor;
    public GameObject lockedDoor;
    // Start is called before the first frame update

    Vector3 emptyStairWellStartingPosition;
    Vector3 startHallStartingPosition;
    Vector3 endHallStartingPosition;
    Vector3 entryHallStartingPosition;

    void Start()
    {
        emptyStairWellStartingPosition = EmptyStairWell.transform.position;
        startHallStartingPosition = StartHall.transform.position;
        entryHallStartingPosition = EntryHall.transform.position;
        endHallStartingPosition = EndHall.transform.position;
    }

    // Update is called once per frame
    void Update()
    {

    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.tag == "Player")
        {
            EmptyStairWell.transform.position = emptyStairWellStartingPosition - ((puzzleController.currentLevel + 1) * new Vector3(0, 3f * 5f, 0));
            StartHall.transform.position = startHallStartingPosition - ((puzzleController.currentLevel + 1) * new Vector3(0, 3f * 5f, 0));
            EntryHall.transform.position = entryHallStartingPosition - ((puzzleController.currentLevel + 1) * new Vector3(0, 3f * 5f, 0));
            EndHall.transform.position = endHallStartingPosition - ((puzzleController.currentLevel + 1) * new Vector3(0, 3f * 5f, 0));
            
            StartHall.SetActive(false);
            floorTransferObject.GetComponent<TransformFloors>().enabled = true;

            unlockedDoor.SetActive(false);
            lockedDoor.SetActive(true);
        }
    }
}
