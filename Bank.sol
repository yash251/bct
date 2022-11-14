//SPDX-License-Identifier: MIT
pragma solidity ^0.8.7;

contract BankAccount {
    address public owner;
    uint256 public balance;

    constructor() {
        owner = msg.sender;
    }

    receive() external payable {
        balance += msg.value;
    }

    function withdraw(uint256 amount, address payable destAddress) public {
        require(msg.sender == owner, "Only owner can withdraw");
        require(amount <= balance, "Insufficent funds");

        destAddress.transfer(amount); // transfer is inbuilt function
        balance -= amount;
    }
}
