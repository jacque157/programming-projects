using UnityEngine;
using UnityEditor;
using System.Collections.Generic;

[CustomEditor(typeof(SliderPuzzle))]
public class CreateCellStartGoalPositions : Editor
{
    //no need for this
    //private SerializedProperty transformsList;
    //private SerializedProperty firstListPositions;
    //private SerializedProperty secondListPositions;
    /*
    private void OnEnable()
    {
        transformsList = serializedObject.FindProperty("transforms");
        firstListPositions = serializedObject.FindProperty("startPos");
        secondListPositions = serializedObject.FindProperty("goalPos");
    }
*/
    public override void OnInspectorGUI()
    {
        serializedObject.Update();
        DrawDefaultInspector();
        SliderPuzzle myTargetScript = (SliderPuzzle)target;

        //EditorGUILayout.PropertyField(transformsList, new GUIContent("Transforms List"), true);

        if (GUILayout.Button("Create Positions for Start Pos(scrambled image)"))
        {
            myTargetScript.startPos.Clear();
            foreach (Transform transform in myTargetScript.transforms)
            {
                myTargetScript.startPos.Add(transform.localPosition);
             
            }

            
        }

        if (GUILayout.Button("Create Positions for Goal Pos(correct image)"))
        {
            myTargetScript.goalPos.Clear();

            foreach (Transform transform in myTargetScript.transforms)
            {
                myTargetScript.goalPos.Add(transform.localPosition);
             
            }
        }
        if (GUILayout.Button("Fill image segments"))
        {
            myTargetScript.goalPos.Clear();

            //assign them to list instantly
            myTargetScript.sliderCellsHolder.GetComponentsInChildren(myTargetScript.slidePuzzleCells);

            myTargetScript.spriteRenderers = myTargetScript.sliderCellsHolder.GetComponentsInChildren<SpriteRenderer>();
            for(int i = 0; i<myTargetScript.spriteRenderers.Length; i++)
            {
                myTargetScript.spriteRenderers[i].sprite = myTargetScript.sprites[i];
            }
        }

        //EditorGUILayout.PropertyField(firstListPositions, new GUIContent("First List Positions"), true);
        //EditorGUILayout.PropertyField(secondListPositions, new GUIContent("Second List Positions"), true);

        serializedObject.ApplyModifiedProperties();
    }
}

