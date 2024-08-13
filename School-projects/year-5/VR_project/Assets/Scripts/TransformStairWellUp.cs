using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TransformStairWellUp : MonoBehaviour
{
    public TorchPuzzle puzzleController;
    public GameObject EmptyStairWell;
    public GameObject StartHall;
    public GameObject EndHall;
    public GameObject EntryHall;

    public GameObject floorTransferObject;
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
            EmptyStairWell.transform.position = emptyStairWellStartingPosition - (puzzleController.currentLevel * new Vector3(0, 3f * 5f, 0));
            StartHall.transform.position = startHallStartingPosition -(puzzleController.currentLevel * new Vector3(0, 3f * 5f, 0));
            EntryHall.transform.position = entryHallStartingPosition - (puzzleController.currentLevel * new Vector3(0, 3f * 5f, 0));
            EndHall.transform.position = endHallStartingPosition - (puzzleController.currentLevel * new Vector3(0, 3f * 5f, 0));

            
            StartHall.SetActive(true);
            floorTransferObject.GetComponent<TransformFloors>().enabled = false;
        }
    }
}
