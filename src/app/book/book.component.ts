import { Component, OnInit } from '@angular/core';

import lifeScience from '../../assets/json_objects/auth_to_abs.json';
import authors from '../../assets/json_objects/list_of_author.json';
import department_to_auth from '../../assets/json_objects/department_to_auth.json';
import department from '../../assets/json_objects/departments.json';
import {AutocompleteLibModule} from 'angular-ng-autocomplete';

@Component({
  selector: 'app-book',
  templateUrl: './book.component.html',
  styleUrls: ['./book.component.css']
})
export class BookComponent{
  

  LS: any;
  search: string;
  dataAuthor: any[];
  dataDepartment: any[];
  keyword: string;
  depKeyword: string;
  name: string;
  abstracts: any[];
  authInDepartment: any[];
  depart: any[];
  department_to_auth: any[];
  chapter: string;
  constructor() {
    this.chapter = 'chapter1';
   
    this.LS = lifeScience;
    this.keyword = 'name';
    this.depKeyword = 'department';
    this.dataAuthor = authors;
    this.dataDepartment = department;
    this.name = "";
    this.abstracts = [];
    this.authInDepartment = [];
    this.search = "author";
    this.department_to_auth = department_to_auth;
    
  }
  
  createPost(name) {
    this.name = name;
    for (let entry of this.LS) {
      let value = entry.name;
      if (value == this.name) {
        this.abstracts = entry.abstracts;
      }  
    }
  }

  getAuthorInDep(dep) {
    for (let entry of this.department_to_auth) {
      let depVal = entry.department;
      if (depVal == dep) {
        this.authInDepartment = entry.name;
      }
    }
  }

  selectEvent(item) {
    // do something with selected item
  }
 
  onChangeSearch(val: string) {
    // fetch remote data from here
    // And reassign the 'data' which is binded to 'data' property.
  }
  
  onFocused(e){
    // do something when input is focused
  }


}
