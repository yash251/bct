//SPDX-License-Identifier: MIT
pragma solidity ^0.8.7;

contract StudentData {
    event Log(string func, address sender, uint256 value, bytes data); // used to log events
    struct Student {
        // struct is a custom data type
        uint256 id;
        uint256 marks;
        string name;
    }
    Student[] students; // array of students
    uint256 nextId = 0;

    function Create(uint256 marks, string memory name) public {
        students.push(Student(nextId, marks, name));
        nextId++;
    }

    function Read(uint256 id)
        public
        view
        returns (
            uint256 uid,
            uint256 marks,
            string memory name
        )
    {
        for (uint256 i = 0; i < students.length; i++) {
            if (students[i].id == id) {
                return (students[i].id, students[i].marks, students[i].name);
            }
        }
    }

    function Update(
        uint256 id,
        uint256 marks,
        string memory name
    ) public {
        for (uint256 i = 0; i < students.length; i++) {
            if (students[i].id == id) {
                students[i].name = name;
                students[i].marks = marks;
            }
        }
    }

    function Delete(uint256 id) public {
        delete students[id];
    }

    fallback() external payable {
        emit Log("fallback", msg.sender, msg.value, msg.data); // fires an event
    }

    receive() external payable {}
}
